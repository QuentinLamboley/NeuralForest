#!/usr/bin/env Rscript
# NCEAS disdat public species distribution benchmark input preparation.
# This operation downloads/install-loads only the openly published
# 'disdat' R data package; no private source datasets or git secrets.
#
# Training is ONLY presence records plus available background. Separate
# surveyed presence/absence outcomes are written to test-label files
# which no R/Python fitting step is allowed to pass to the estimator.
# Species selected by PREDECLARED TRAINING presence count, never by
# their independent test performance or surveyed evaluation labels.
suppressPackageStartupMessages(library(disdat))
suppressPackageStartupMessages(library(jsonlite))
args<-commandArgs(trailingOnly=TRUE)
if(length(args)!=3L)stop(
 "Usage: po_nceas_export.R REGION MAX_SPECIES OUTPUT_DIR")
region<-args[[1]]
if(!(region %in% c("AWT","CAN","NSW","NZ","SA","SWI")))
  stop("Unknown NCEAS region")
limit<-as.integer(args[[2]])
if(!is.finite(limit)||limit<1||limit>226)
  stop("Invalid prespecified number of species")
out<-args[[3]]
dir.create(out,recursive=TRUE,showWarnings=FALSE)
po<-disdat::disPo(region)
bg<-disdat::disBg(region)
predictors<-as.character(disdat::disPredictors(region))
if(!all(c("spid","group") %in% names(po)) ||
   !all(predictors %in% names(po)) ||
   !all(predictors %in% names(bg)))
  stop("Unexpected package disdat region schema")
# Deterministic predeclared inclusion criterion: most training PO records
# among species with at least 40 usable PO observations.
score<-sort(table(as.character(po$spid)),decreasing=TRUE)
eligible<-names(score)[as.integer(score)>=40L]
if(length(eligible)<1L)stop("No training-supported NCEAS species")
selected<-head(eligible,limit)
# Draw a fixed-sized common region background using training-only
# information; no surveyed outcome enters the random number generator.
set.seed(87017L)
if(nrow(bg)>2500L)
 bg<-bg[sort(sample(seq_len(nrow(bg)),2500L)),,drop=FALSE]
bg<-bg[stats::complete.cases(bg[,predictors,drop=FALSE]),,drop=FALSE]
if(nrow(bg)<100L)stop("Insufficient usable environmental availability")
species_manifest<-list()
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
    stop(paste("Independent evaluation schema mismatch",s))
  # Site ID and coordinates, when shared, must match in row order.
  for(key in intersect(c("site","siteid","x","y"),intersect(names(env),names(pa)))) {
    a<-env[[key]];b<-pa[[key]]
    if(is.numeric(a)&&is.numeric(b)) {
      if(!isTRUE(all.equal(a,b,tolerance=1e-6,check.attributes=FALSE)))
        stop(paste("Evaluation sites misaligned for",s,key))
    } else if(!identical(as.character(a),as.character(b))) {
      stop(paste("Evaluation sites misaligned for",s,key))
    }
  }
  p<-obs[stats::complete.cases(obs[,predictors,drop=FALSE]),
        predictors,drop=FALSE]
  labels<-as.character(pa[[s]])
  test_ok<-stats::complete.cases(env[,predictors,drop=FALSE])&
           !is.na(labels) & labels %in% c("0","1")
  test<-env[test_ok,predictors,drop=FALSE]
  truth<-as.integer(labels[test_ok])
  if(nrow(p)<30L || nrow(test)<25L)
    stop(paste("Insufficient training/test predictors",s))
  dir<-file.path(out,s)
  dir.create(dir,recursive=TRUE,showWarnings=FALSE)
  utils::write.csv(p,file.path(dir,"presence_train.csv"),
                   row.names=FALSE)
  utils::write.csv(bg[,predictors,drop=FALSE],
                   file.path(dir,"background_train.csv"),
                   row.names=FALSE)
  utils::write.csv(test,file.path(dir,"survey_predictors.csv"),
                   row.names=FALSE)
  # TEST LABELS are never passed to a model fit, only the audit reader
  # after all model predictions have been recorded.
  utils::write.csv(data.frame(survey_outcome=truth),
                   file.path(dir,"survey_outcomes_DO_NOT_TRAIN.csv"),
                   row.names=FALSE)
  species_manifest[[length(species_manifest)+1L]]<-list(
     id=s,group=group,n_p_training=nrow(p),
     n_background=nrow(bg),
     n_survey_test=nrow(test),
     n_survey_true_present=sum(truth==1L),
     n_survey_true_absent=sum(truth==0L),
     n_missing_test_labels=sum(!test_ok))
}
jsonlite::write_json(list(region=region,selected_by=
 "training_PO_count_only",species=species_manifest,
 predictor_columns=predictors,
 background_policy="common region disdat background subsampled 2500",
 independent_survey_absences_used_for_training=FALSE,
 original_reference_doi="10.17161/bi.v15i2.13384",
 notes="Test survey outcomes are NEVER allowed to select species by accuracy or tune model parameters."
),file.path(out,"input_manifest.json"),
 pretty=TRUE,auto_unbox=TRUE,digits=12)
print(paste("Prepared",length(species_manifest),
            "independent-survey NCEAS species in",region))
