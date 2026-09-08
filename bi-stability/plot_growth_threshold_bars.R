#!/usr/bin/env Rscript
# Plot current scan endpoints; do not rerun simulations or read live parameters.
suppressPackageStartupMessages({library(ggplot2); library(grid)})
script_arg <- grep('^--file=', commandArgs(FALSE), value=TRUE)
here <- dirname(normalizePath(sub('^--file=', '', script_arg)))
args <- commandArgs(TRUE)
output <- if(length(args)) normalizePath(args[[1]]) else file.path(here, 'output')
stage <- if(length(args)>1) args[[2]] else 'final'
lag_mode <- if(length(args)>2) args[[3]] else 'on'
focal_species <- if(length(args)>3) args[[4]] else 'Col'
stopifnot(focal_species %in% c('Col','Cs'))
stopifnot(stage %in% c('final', 'baseline_20_cycles'), lag_mode %in% c('on','off'))
figdir <- file.path(output, 'figures')
dir.create(figdir, recursive=TRUE, showWarnings=FALSE)
read <- function(path) read.csv(path, check.names=FALSE)
is_true <- function(x) tolower(as.character(x)) %in% c('true','1')
paths <- c(file.path(output,'pairwise',stage,'run_summary.csv'),
           file.path(output,'community',stage,'run_summary.csv'),
           file.path(output,'growth_rate_axis_parameters.csv'))
a <- read(paths[1]); b <- read(paths[2]); coefficients <- read(paths[3])
a <- a[a$species %in% c(focal_species,'Mi'), ]; b$species <- 'All NG'
cols <- c('run_id','species','ratio','lag_mode','initial_condition',
          'extinction_mode','cycle_end_bt','converged_last_n','cycle')
a <- a[,cols]; b <- b[,cols]
a$source_kind <- 'pairwise'; b$source_kind <- 'community'
points <- rbind(a,b)
points <- points[points$lag_mode==lag_mode & points$extinction_mode=='hard_extinction', ]
points$converged_last_n <- is_true(points$converged_last_n)
points$outcome <- factor(ifelse(points$cycle_end_bt < 1e-4, 'Bt extinction', 'Bt survives'),
                         levels=c('Bt survives','Bt extinction'))
spec <- data.frame(species=c(focal_species,'Mi','All NG'),
  dm_slope=c(coefficients$dm_growth_slope[match(c(focal_species,'Mi'),coefficients$species)],
             sum(coefficients$dm_growth_slope)),
  bt_spent_intercept=c(coefficients$bt_spent_growth_intercept[match(c(focal_species,'Mi'),coefficients$species)],
                       sum(coefficients$bt_spent_growth_intercept)))
stopifnot(nrow(coefficients)==10, !anyNA(spec), nrow(points)==600)
bars <- data.frame(species=rep(c(focal_species,'Mi','All NG'),each=2),
  initial_condition=rep(c('ng_low_bt_high','ng_high_bt_low'),3),
  x=c(0,.82,2.1,2.92,4.2,5.02),
  label=paste(rep(c(focal_species,'Mi','All NG'),each=2),rep(c('1:1000','1000:1'),3),sep='\n'))
points <- merge(merge(points,spec,by='species',sort=FALSE),bars,
                by=c('species','initial_condition'),sort=FALSE)
points$initial_growth_rate <- points$bt_spent_intercept + points$dm_slope * points$ratio
points$growth_definition <- ifelse(points$species=='All NG',
  'sum of 10 species potential initial per-capita growth rates',
  'species potential initial per-capita growth rate')
points$stage <- stage
points <- points[order(points$x,points$ratio), ]
stopifnot(!anyDuplicated(paste(points$x,points$ratio)))
# Nearest scanned point defines each interval; outer edges stay within actual
# scan support. Transition edges are display midpoints, not inferred thresholds.
cells <- do.call(rbind,lapply(split(points,points$x),function(g){
  g <- g[order(g$initial_growth_rate), ]; y <- g$initial_growth_rate
  mid <- (y[-length(y)]+y[-1])/2
  g$ymin <- pmax(0,c(y[1],mid)); g$ymax <- pmin(3,c(mid,tail(y,1)))
  g[g$ymax>g$ymin, ]
}))
rownames(cells) <- NULL
cells$interval_source <- 'simulation scan'
# User-requested blue extension from zero to the first actual scan point.
# Keep assumed display intervals separate from simulated observations.
zero_extension <- points[!duplicated(points$x), ]
zero_extension$ymin <- 0
zero_extension$ymax <- pmin(3,zero_extension$initial_growth_rate)
zero_extension$outcome <- factor('Bt survives',levels=levels(points$outcome))
zero_extension$interval_source <- 'user-assumed Bt survival below scan support'
zero_extension[,c('run_id','ratio','cycle_end_bt','converged_last_n','cycle',
                 'initial_growth_rate')] <- NA
cells <- rbind(cells,zero_extension[zero_extension$ymax>0, ])
bt_mu <- unique(coefficients$bt_initial_growth_h_inv)
stopifnot(length(bt_mu)==1, is.finite(bt_mu))
stem <- paste0(tolower(focal_species), '_mi_bt_extinction_threshold_bars_initial_growth_0_3_sum_10ng',
               '_bt_dashed_line_arial_unified_6x3in_illustrator_safe',
               if(lag_mode=='off') '_lag_off' else '',
               if(stage!='final') paste0('_',stage) else '')
width <- .54
p <- ggplot(cells,aes(x=x,fill=outcome)) +
  geom_rect(aes(xmin=x-width/2,xmax=x+width/2,ymin=ymin,ymax=ymax),colour=NA) +
  geom_rect(data=bars,aes(xmin=x-width/2,xmax=x+width/2,ymin=0,ymax=3),
            inherit.aes=FALSE,fill=NA,colour='#111111',linewidth=.35) +
  geom_hline(yintercept=bt_mu,linetype='dashed',linewidth=.45,colour='#111111') +
  geom_point(data=data.frame(x=-.36,y=bt_mu),aes(x=x,y=y),inherit.aes=FALSE,
             shape=8,size=3.5,stroke=.8,colour='#111111') +
  geom_point(data=points[!points$converged_last_n & points$initial_growth_rate<=3,],
             aes(x=x,y=initial_growth_rate),inherit.aes=FALSE,shape=4,size=1.5,stroke=.35) +
  geom_point(data=data.frame(x=0,y=-1,key='Bt indicator'),aes(x=x,y=y,shape=key),
             inherit.aes=FALSE,alpha=0,show.legend=TRUE) +
  scale_fill_manual(values=c('Bt survives'='#6BAED6','Bt extinction'='#E77C73'),drop=FALSE) +
  scale_shape_manual(values=c('Bt indicator'=8),labels=c('Bt indicator'='= Bt growth rate'),name=NULL) +
  scale_x_continuous(breaks=bars$x,labels=bars$label,limits=c(-.50,5.52),expand=expansion(mult=0)) +
  scale_y_continuous(breaks=0:3,expand=expansion(mult=0)) +
  coord_cartesian(ylim=c(0,3),clip='off') +
  labs(x='Initial NG:Bt ratio',y=expression(paste('Initial growth rate, ',plain('μ')^plain('0'),
       ' (',plain('h')^plain('-1'),')')),fill=NULL) +
  guides(fill=guide_legend(order=1,direction='horizontal',keywidth=unit(16,'pt'),keyheight=unit(8,'pt')),
         shape=guide_legend(order=2,direction='horizontal',override.aes=list(alpha=1,colour='#111111',size=3.5))) +
  theme_classic(base_family='Arial',base_size=10) +
  theme(text=element_text(family='Arial',size=10,colour='#111111'),
        axis.title=element_text(size=10),axis.text.x=element_text(size=10,lineheight=.95,colour='#111111'),
        axis.text.y=element_text(size=8,colour='#111111'),
        axis.ticks=element_line(linewidth=.35,colour='#111111'),axis.ticks.x=element_blank(),
        axis.line=element_line(linewidth=.45,colour='#111111'),
        legend.position='top',legend.justification='left',legend.box.just='left',
        legend.box='horizontal',legend.direction='horizontal',legend.text=element_text(size=8),
        legend.margin=margin(t=0,r=0,b=2,l=0,unit='pt'),legend.box.margin=margin(0,0,0,0),
        plot.margin=margin(t=12,r=8,b=8,l=5,unit='pt'))
svglite::svglite(file.path(figdir,paste0(stem,'.svg')),width=6,height=3,system_fonts=list(sans='Arial'))
print(p); dev.off()
cairo_pdf(file.path(figdir,paste0(stem,'.pdf')),width=6,height=3,family='Arial')
print(p); dev.off()
preview <- Sys.getenv('BT_SCAN_PREVIEW')
if(nzchar(preview)){
  dir.create(preview,recursive=TRUE,showWarnings=FALSE)
  ragg::agg_png(file.path(preview,paste0(stem,'.png')),width=6,height=3,units='in',res=200)
  print(p); dev.off()
}
cat('Saved:',file.path(figdir,stem),'\n')
