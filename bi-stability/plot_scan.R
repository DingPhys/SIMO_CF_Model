#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(ggplot2);library(grid)})
arg <- grep('^--file=', commandArgs(FALSE), value=TRUE)
here <- dirname(normalizePath(sub('^--file=', '', arg)))
cli <- commandArgs(TRUE)
output <- if (length(cli)) normalizePath(cli[[1]]) else file.path(here, 'output')
figdir <- file.path(output,'figures');dir.create(figdir,showWarnings=FALSE)
preview <- Sys.getenv('BT_SCAN_PREVIEW')
if(nzchar(preview)) dir.create(preview,recursive=TRUE,showWarnings=FALSE)
read <- function(p) read.csv(p,check.names=FALSE)
ratio <- read(file.path(output,'ratio_grid.csv'))
sp <- c('Col','Cs','Et','Eu.c','Eu.l','Im','Ld','Lsp','Mi','Va','All NG')
base_theme <- theme_classic(base_family='Arial',base_size=8) + theme(
 axis.title=element_text(size=10),axis.text=element_text(size=8),
 strip.text=element_text(size=8),legend.text=element_text(size=8),legend.title=element_text(size=8),
 plot.title=element_text(size=10),plot.subtitle=element_text(size=8),
 strip.background=element_rect(fill='white',colour=NA),
 axis.line=element_line(linewidth=.18),axis.ticks=element_line(linewidth=.18),
 panel.spacing.x=unit(12,'pt'))
save_plot <- function(p,stem,w,h){
 svglite::svglite(file.path(figdir,paste0(stem,'.svg')),width=w,height=h);print(p);dev.off()
 cairo_pdf(file.path(figdir,paste0(stem,'.pdf')),width=w,height=h,family='Arial');print(p);dev.off()
 if(nzchar(preview)){ragg::agg_png(file.path(preview,paste0(stem,'.png')),width=w,height=h,units='in',res=160);print(p);dev.off()}
}
combined <- do.call(rbind,lapply(c('baseline_20_cycles','final'),function(stage){
 do.call(rbind,lapply(c('pairwise','community'),function(kind){
  d <- read(file.path(output,kind,stage,'initial_condition_comparison.csv'))
  d <- d[,c('species','ratio','lag_mode','extinction_mode','max_endpoint_log10_difference','both_initial_conditions_converged','survival_class_differs','relative_composition_l1')]
  d$stage <- if(stage=='baseline_20_cycles') 'After 20 cycles' else 'After extension'
  d
 }))
}))
combined$scan_index <- match(round(combined$ratio,12),round(ratio$ratio,12))-1
combined$both_initial_conditions_converged <- tolower(as.character(combined$both_initial_conditions_converged)) %in% c('true','1')
stopifnot(!anyNA(combined$scan_index))
combined$species <- factor(combined$species,levels=rev(sp))
combined$condition <- paste(ifelse(combined$lag_mode=='on','Lag on','Lag off'),
 ifelse(combined$extinction_mode=='hard_extinction','/ endpoint deletion','/ no deletion'))
combined$stage <- factor(combined$stage,levels=c('After 20 cycles','After extension'))
breaks <- c(0,26,50,75,99);labels <- c('0','~0.1','1','~10','100')
for(masked in c(FALSE,TRUE)){
 d <- combined;d$display <- pmin(d$max_endpoint_log10_difference,6)
 if(masked)d$display[!d$both_initial_conditions_converged] <- NA
 p <- ggplot(d,aes(scan_index,species,fill=display))+geom_tile()+
 facet_grid(stage~condition)+scale_fill_gradient(low='white',high='#B2182B',limits=c(0,6),na.value='#D0D0D0',name='Endpoint difference\n(max |log10 fold change|)',guide=guide_colourbar(display='rectangles'))+
 scale_x_continuous(breaks=breaks,labels=labels,expand=c(0,0))+scale_y_discrete(expand=c(0,0))+
 labs(x='Scanned consumption-rate ratio: NG on DM / NG on Bt products',y=NULL,
 subtitle=if(masked)'Grey: at least one initial-condition trajectory has not converged' else 'Finite-time differences alone do not establish bistability')+
 base_theme+theme(axis.line=element_blank(),legend.position='bottom')
 save_plot(p,if(masked)'initial_condition_difference_converged' else 'initial_condition_difference_all',10,6.2)
}
# Endpoint outcomes: no hand-added extrapolated zero-growth observations.
bar_data <- do.call(rbind,lapply(c('baseline_20_cycles','final'),function(stage){
 a<-read(file.path(output,'pairwise',stage,'run_summary.csv'));a<-a[a$species %in% c('Cs','Mi'),]
 b<-read(file.path(output,'community',stage,'run_summary.csv'));b$species<-'All NG'
 cols<-c('species','ratio','lag_mode','initial_condition','extinction_mode','cycle_end_bt','converged_last_n','cycle')
 d<-rbind(a[,cols],b[,cols]);d<-d[d$extinction_mode=='hard_extinction',]
 d$stage<-if(stage=='baseline_20_cycles')'After 20 cycles' else 'After extension';d
}))
bar_data$scan_index<-match(round(bar_data$ratio,12),round(ratio$ratio,12))-1
bar_data$converged_last_n<-tolower(as.character(bar_data$converged_last_n)) %in% c('true','1')
bar_data$outcome<-ifelse(bar_data$cycle_end_bt<1e-4,'Bt below threshold','Bt survives')
bar_data$initial<-ifelse(bar_data$initial_condition=='ng_low_bt_high','NG:Bt = 1:1000','NG:Bt = 1000:1')
bar_data$species<-factor(bar_data$species,levels=c('Cs','Mi','All NG'))
bar_data$stage<-factor(bar_data$stage,levels=c('After 20 cycles','After extension'))
for(lagmode in c('on','off')){
 d<-bar_data[bar_data$lag_mode==lagmode,]
 p<-ggplot(d,aes(scan_index,initial,fill=outcome))+geom_tile(height=.7)+
 geom_point(data=d[!d$converged_last_n,],shape=4,size=.65,stroke=.2,show.legend=FALSE)+
 facet_grid(stage~species)+scale_fill_manual(values=c('Bt below threshold'='#E77C73','Bt survives'='#6BAED6'),drop=FALSE)+
 scale_x_continuous(breaks=breaks,labels=labels,expand=c(0,0))+
 labs(x='Scanned consumption-rate ratio: NG on DM / NG on Bt products',y=NULL,fill=NULL,
 subtitle=paste0('Lag ',lagmode,'; x marks unconverged trajectories. Initial NG:Bt uses total NG biomass.'))+
 base_theme+theme(legend.position='bottom',axis.line.y=element_blank(),axis.ticks.y=element_blank())
 save_plot(p,paste0('bt_outcomes_lag_',lagmode),8,3.5)
}
cat('Saved SVG/PDF in',figdir,'\n')
