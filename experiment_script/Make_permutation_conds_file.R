# Make condition trials for randomization for Templeton Exp 2A

#Three variables: change/no change, retro-cue/post-cue, short/long retention interval
#In the main experiment, we want 80% post-cue trials at the long retention interval (40% change, 40% no change),
#10% retro-cue trials at the short retention interval (5% change, 5% no change), 10% retro-cue trials at the long
#retention interval (5% change, 5% no change)
#Using the constrained location approach

#library(randtests)
nTrials = 50 #trials per fMRI run
all_perms = read.csv("limited_perms.csv")
for (i in 1:15){
  rand_rows = round(runif(nTrials,1,nrow(all_perms)))

  trials = all_perms[rand_rows,]

  change_1_nochange_0 = c(rep(0,0.5*nTrials),rep(1,0.5*nTrials)) #50/50 chance
  retro_0_post_1 = c(rep(0,0.2*nTrials),rep(1,0.8*nTrials))
  short_0_long_1 = c(rep(0,0.5*0.2*nTrials),rep(1,0.5*0.2*nTrials),rep(1,0.8*nTrials))
  trials = cbind(trials,retro_0_post_1,change_1_nochange_0,short_0_long_1)
  trials = trials[sample(nrow(trials)),]

  colnames(trials) = c("Used_image_1","Used_image_2","Used_image_3","Used_image_4","Used_image_5","Used_image_6","Used_image_7","Used_image_8","Unused_image_1","Unused_image_2","Retro_post_0_1","Change_yes_no","Short_long_0_1")
  write.csv(trials,paste0("limited_perms_conds_",i,".csv"),row.names = F)
}