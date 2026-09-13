#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(ggplot2)
  library(grid)
})
arg <- grep('^--file=', commandArgs(FALSE), value=TRUE)
here <- dirname(normalizePath(sub('^--file=', '', arg)))
cli <- commandArgs(TRUE)
output <- if (length(cli)) normalizePath(cli[[1]]) else file.path(here, 'simulation_results')
figdir <- file.path(output,'figures')
dir.create(figdir,showWarnings=FALSE)
read <- function(p) read.csv(p,check.names=FALSE)
ratio <- read(file.path(output,'ratio_grid.csv'))
base_theme <- theme_classic(base_family='Arial',base_size=8) + theme(
 axis.title=element_text(size=10),axis.text=element_text(size=8),
 strip.text=element_text(size=8),legend.text=element_text(size=8),legend.title=element_text(size=8),
 plot.title=element_text(size=10),plot.subtitle=element_text(size=8),
 strip.background=element_rect(fill='white',colour=NA),
 axis.line=element_line(linewidth=.18),axis.ticks=element_line(linewidth=.18),
 panel.spacing.x=unit(12,'pt'))
save_plot <- function(p,stem,w,h){
 svglite::svglite(file.path(figdir,paste0(stem,'.svg')),width=w,height=h)
 print(p)
 dev.off()
 cairo_pdf(file.path(figdir,paste0(stem,'.pdf')),width=w,height=h,family='Arial')
 print(p)
 dev.off()
}
a <- read(file.path(output,'run_summary_pairwise.csv'))
a <- a[a$species %in% c('Col','Mi'),]
b <- read(file.path(output,'run_summary_community.csv'))
b$species <- 'All NG'
cols <- c('species','ratio','lag_mode','initial_condition','extinction_mode','cycle_end_bt','converged_last_n','cycle')
bar_data <- rbind(a[,cols],b[,cols])
bar_data <- bar_data[bar_data$extinction_mode=='hard_extinction',]
breaks <- c(0,26,50,75,99)
labels <- c('0','~0.1','1','~10','100')
bar_data$scan_index<-match(round(bar_data$ratio,12),round(ratio$ratio,12))-1
bar_data$converged_last_n<-tolower(as.character(bar_data$converged_last_n)) %in% c('true','1')
bar_data$outcome<-ifelse(bar_data$cycle_end_bt<1e-4,'Bt below threshold','Bt survives')
bar_data$initial<-ifelse(bar_data$initial_condition=='ng_low_bt_high','NG:Bt = 1:1000','NG:Bt = 1000:1')
bar_data$species<-factor(bar_data$species,levels=c('Col','Mi','All NG'))
for(lagmode in 'on'){
 d<-bar_data[bar_data$lag_mode==lagmode,]
 p<-ggplot(d,aes(scan_index,initial,fill=outcome))+geom_tile(height=.7)+
 geom_point(data=d[!d$converged_last_n,],shape=4,size=.65,stroke=.2,show.legend=FALSE)+
 facet_grid(.~species)+scale_fill_manual(values=c('Bt below threshold'='#E77C73','Bt survives'='#6BAED6'),drop=FALSE)+
 scale_x_continuous(breaks=breaks,labels=labels,expand=expansion(mult=c(.015,.04)))+
 labs(x='Scanned consumption-rate ratio: NG on DM / NG on Bt products',y=NULL,fill=NULL,
 subtitle=paste0('Lag ',lagmode,'; x marks unconverged trajectories. Initial NG:Bt uses total NG biomass.'))+
 base_theme+theme(legend.position='bottom',axis.line.y=element_blank(),axis.ticks.y=element_blank())
 save_plot(p,paste0('bt_outcomes_lag_',lagmode),8,2.2)
}
cat('Saved SVG/PDF in',figdir,'\n')
