#!/usr/bin/env Rscript
# NCEAS six-region presence-only export for source-independent survey tests.
# Fits NEVER receive survey outcomes. Species choice, predictor coding,
# background subsampling and inner spatial bands use TRAINING data only.
suppressPackageStartupMessages(library(disdat))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=4L)stop(
 "Usage: po_nceas_multi_export.R REGION MAX_SPECIES MIN_PRESENCES OUTDIR")
region<-args[[1]]
max_sp<-as.integer(args[[2]])
min_po<-as.integer(args[[3]])
out<-args[[4]]
if(!region %in% c("AWT","CAN","NSW","NZ","SA","SWI"))
  stop("Unknown region")
if(!is.finite(max_sp)||max_sp<1L||max_sp>226L||
   !is.finite(min_po)||min_po<70L||min_po>10000L)
  stop("Invalid predeclared training-only species selection")
dir.create(out,showWarnings=FALSE,recursive=TRUE)
po<-disdat::disPo(region)
bg<-disdat::disBg(region)
predictors<-as.character(disdat::disPredictors(region))
if(!all(c("spid","group","x","y") %in% names(po))||
   !all(c("x","y") %in% names(bg))||
   !all(predictors %in% names(po))||
   !all(predictors %in% names(bg)))
   stop("Unexpected disdat training schema")

# These are *documented* categorical predictors (disdat 1.1-0 manual).
# Levels are learned exclusively from training PO and availability,
# never from independent survey features or observed PA outcomes.
categorical<-intersect(c("ontveg","vegsys","soilfert","age",
                          "toxicats","calc"),predictors)
cat_levels<-setNames(lapply(categorical,function(name)
  sort(unique(c(as.character(po[[name]]),
                as.character(bg[[name]]))))),categorical)
cat_levels<-lapply(cat_levels,function(v)v[!is.na(v)&v!="NA"])
for(name in categorical)
 if(length(cat_levels[[name]])<2L||length(cat_levels[[name]])>50L)
   stop(paste("Unexpected training categorical support",name))
feature_names<-character(0)
for(name in predictors) {
  if(name %in% categorical)
    feature_names<-c(feature_names,
                     paste0(name,"_cat_",seq_along(cat_levels[[name]])))
  else feature_names<-c(feature_names,name)
}
if(anyDuplicated(feature_names))stop("Duplicate encoded columns")
encode<-function(raw) {
  out<-vector("list",length(feature_names))
  names(out)<-feature_names
  for(name in predictors) {
    if(name %in% categorical) {
      actual<-as.character(raw[[name]])
      for(k in seq_along(cat_levels[[name]]))
        out[[paste0(name,"_cat_",k)]]<-
          as.numeric(actual==cat_levels[[name]][[k]])
    } else {
      v<-raw[[name]]
      if(is.factor(v))v<-as.character(v)
      out[[name]]<-as.numeric(v)
    }
  }
  d<-as.data.frame(out,check.names=FALSE)
  if(!identical(names(d),feature_names))
    stop("Encoding altered column names")
  d
}

# Common background is sampled BEFORE species/validation selection,
# with a fixed region seed, not reselected by any surveyed outcome.
set.seed(87017L)
bg_ix<-if(nrow(bg)>2500L)
 sort(sample(seq_len(nrow(bg)),2500L)) else seq_len(nrow(bg))
bg<-bg[bg_ix,,drop=FALSE]
bg<-bg[stats::complete.cases(bg[,predictors,drop=FALSE])&
         is.finite(bg$x)&is.finite(bg$y),,drop=FALSE]
if(nrow(bg)<100L)stop("Insufficient region availability")
background_encoded<-encode(bg)
if(any(!is.finite(as.matrix(background_encoded))))
  stop("Nonfinite encoded region background")

present_ok<-stats::complete.cases(po[,predictors,drop=FALSE])&
  is.finite(po$x)&is.finite(po$y)&!is.na(po$spid)
po<-po[present_ok,,drop=FALSE]
counts<-sort(table(as.character(po$spid)),decreasing=TRUE)
eligible<-names(counts)[as.integer(counts)>=min_po]
selected<-head(eligible,max_sp)
input_info<-list()
for(s in selected) {
 obs<-po[as.character(po$spid)==s,,drop=FALSE]
 group<-as.character(obs$group[[1L]])
 if(region %in% c("AWT","NSW")) {
   env<-disdat::disEnv(region,group)
   pa<-disdat::disPa(region,group)
 } else {
   env<-disdat::disEnv(region)
   pa<-disdat::disPa(region)
 }
 if(!all(predictors %in% names(env))||
    !(s %in% names(pa))||nrow(env)!=nrow(pa))
   stop(paste("Survey schema mismatched",region,s,group))
 for(key in intersect(c("site","siteid","x","y"),
              intersect(names(env),names(pa)))) {
   a<-env[[key]];b<-pa[[key]]
   if(is.numeric(a)&&is.numeric(b)) {
     if(!isTRUE(all.equal(a,b,tolerance=1e-6,
                          check.attributes=FALSE)))
       stop(paste("Survey site alignment mismatch",s,key))
   } else if(!identical(as.character(a),as.character(b)))
     stop(paste("Survey site alignment mismatch",s,key))
 }
 # Survey label NA rows are excluded only from independent scoring,
 # NEVER from training data or choice of predictors/species.
 labels<-as.character(pa[[s]])
 valid<-stats::complete.cases(env[,predictors,drop=FALSE])&
         !is.na(labels)&labels %in% c("0","1")
 test<-env[valid,,drop=FALSE]
 true<-as.integer(labels[valid])
 if(nrow(test)<20L)stop(paste("Too few independent survey records",s))
 test_features<-encode(test)
 raw_train<-encode(obs)
 if(any(!is.finite(as.matrix(raw_train)))||
    any(!is.finite(as.matrix(test_features))))
    stop("Nonfinite encoded training or test covariate")
 name<-file.path(out,s)
 dir.create(name,recursive=TRUE,showWarnings=FALSE)
 utils::write.csv(raw_train,file.path(name,"presence_train.csv"),
                  row.names=FALSE)
 utils::write.csv(background_encoded,
                  file.path(name,"background_train.csv"),row.names=FALSE)
 utils::write.csv(test_features,
                  file.path(name,"survey_predictors.csv"),row.names=FALSE)
 utils::write.csv(obs[,c("x","y"),drop=FALSE],
                  file.path(name,"presence_training_coordinates.csv"),
                  row.names=FALSE)
 utils::write.csv(bg[,c("x","y"),drop=FALSE],
                  file.path(name,"background_training_coordinates.csv"),
                  row.names=FALSE)
 # Isolated audit-only file never passed to model training.
 utils::write.csv(data.frame(survey_outcome=true),
                  file.path(name,"survey_outcomes_AUDIT_ONLY.csv"),
                  row.names=FALSE)
 input_info[[length(input_info)+1L]]<-list(
   id=s,group=group,n_training_presence=nrow(obs),
   n_background=nrow(bg),n_survey_rows=nrow(test),
   coordinate_reference=as.character(disdat::disCRS(region,"EPSG")),
   test_label_file="survey_outcomes_AUDIT_ONLY.csv")
}
jsonlite::write_json(list(region=region,
  selection_rule="highest complete-case PO count >= prespecified minimum",
  minimum_training_presence=min_po,limit=max_sp,
  selected_species=input_info,
  n_available_species_above_minimum=length(eligible),
  n_region_training_presence_rows=nrow(po),
  n_region_background_rows=nrow(bg),
  raw_predictors=predictors,encoded_predictors=feature_names,
  categorical_levels=cat_levels,
  data_package_version=as.character(packageVersion("disdat")),
  test_outcome_used_for_fitting=FALSE,
  survey_outcomes_file_only_used_after_prediction=TRUE,
  note="Training-only preprocessing; x/y used ONLY for inner-fold selection, not as predictor."),
  file.path(out,"manifest.json"),pretty=TRUE,auto_unbox=TRUE)
cat(sprintf("Prepared %d/%d eligible species (%d minimum PO) for %s\n",
  length(selected),length(eligible),min_po,region))
