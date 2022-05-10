
setwd('/Users/dyu/Dropbox/tf-notebooks/cuda_comp/mat_mul_example')

cand_n = seq(1000,3000,by=500)
nc = length(cand_n)
nc
time_float = list()
time_double = list()

for(i in 1:nc)
{
    n = cand_n[i]
    time_float[[i]] = as.matrix(read.table(file=paste0('mm_comp_time_n',n,'_float.txt'),header=T,sep='\t')[,1:6])
    time_double[[i]] = as.matrix(read.table(file=paste0('mm_comp_time_n',n,'_double.txt'),header=T,sep='\t')[,1:6])
}

m_name = c("DLL-Kernel","DLL-cuBLAS", "PyCuda", "Numba", "TensorFlow", "PyTorch")
s_first_mat = matrix(0,5,6)
s_mean_mat = matrix(0,5,6)
s_sd_mat = matrix(0,5,6)
colnames(s_first_mat) = m_name
colnames(s_mean_mat) = m_name
colnames(s_sd_mat) = m_name

d_first_mat = matrix(0,5,6)
d_mean_mat = matrix(0,5,6)
d_sd_mat = matrix(0,5,6)
colnames(d_first_mat) = m_name
colnames(d_mean_mat) = m_name
colnames(d_sd_mat) = m_name

for(i in 1:nc)
{
    s_first_mat[i,] = time_float[[i]][1,]
    s_mean_mat[i,] = apply(time_float[[i]][-1,],2,mean)
    s_sd_mat[i,] = apply(time_float[[i]][-1,],2,sd)/sqrt(10)
    
    d_first_mat[i,] = time_double[[i]][1,]
    d_mean_mat[i,] = apply(time_double[[i]][-1,],2,mean)
    d_sd_mat[i,] = apply(time_double[[i]][-1,],2,sd)/sqrt(10)
}

s_first_mat
s_mean_mat
s_first_mat - s_mean_mat
s_sd_mat

d_first_mat
d_mean_mat
d_first_mat - d_mean_mat
d_sd_mat

s_first_mat - s_mean_mat
d_first_mat - d_mean_mat

diff = rbind(s_first_mat-s_mean_mat, d_first_mat - d_mean_mat)
apply(diff,2,mean)

summ_tex_mat = matrix("",10,8)
colnames(summ_tex_mat) = c("Precision", "$n$", m_name)
summ_tex_mat[1,1] = "\\multirow{5}{*}{Single}"
summ_tex_mat[6,1] = "\\multirow{5}{*}{Double}"
summ_tex_mat[,2] = rep(cand_n,2)
for(i in 1:5)
{
    for(j in 1:6)
    {
        summ_tex_mat[i,j+2] = sprintf("%.4f", s_mean_mat[i,j])
        summ_tex_mat[i+5,j+2] = sprintf("%.4f", d_mean_mat[i,j])
    }
}
summ_tex_mat
summ_tex_mat[,8] = paste(summ_tex_mat[,8],"\\\\")
summ_tex_mat[5,8] = paste(summ_tex_mat[5,8], '\\hline')
write.table(summ_tex_mat, file='summ_mm_comp_time.txt', sep=' & ', col.names = T, quote=F, row.names=F)


fig1 <- t(s_mean_mat)
yl <- c(1e-3,1e0)
yat <- 10^(-3:0)
ylb <- c(expression(10^-3),expression(10^-2),expression(10^-1),expression(10^0))

library(gplots)

library(RColorBrewer)
cand_col = brewer.pal(6, "Set1")
cand_col
#cand_col =  gray(seq(0.4,0.9,length=6))
quartz()
par(mar=c(5,4,2,2))
par(xaxs='i')
par(yaxs='i')
# Example with confidence intervals and grid
mybarcol <- "gray20"
m.name  <- c('1. DLL-Kernel','2. DLL-cuBLAS', '3. PyCuda', '4. Numba','5. TF', '6. PyTorch' )
plot(1:10, type='n',xlim=c(0,36), ylim=yl, axes=F,log='y',xlab='',ylab='',asp=1)
abline(h=yat, lty=3,col=gray(0.2))
par(new=T)
mp <- barplot2(fig1, beside = TRUE, #angle = c(-30,30,45,-45,60),density = 30,
               col = cand_col,xlim=c(0,36), ylim = yl,width=1,log="y",
               xlab='Dimension',ylab='Computation Time (sec.)',#names.arg=c('1','2','3','4','5'),
               main = "", font.main = 4,#c(0.001,0.1,10,1000),
               cex.names = 1, plot.ci = F,cex=1, #grid.inc=10,
               plot.grid =F,axes=F)

mtext(side = 1, at =as.vector(mp), line = 0,
      text = rep(1:6,5), col = "black",cex=0.8)
mtext(side = 1, at = colMeans(mp), line = 1.5,
      text = cand_n, col = "black")
legend('topleft',m.name,fill= cand_col, cex=0.8)#,angle = c(-30,30,45,-45,60),density = 30)
box()
axis(2, at=yat,labels=ylb, col.axis="black", las=1)

w = 640
png(file="mm_example_single.png",width=w, height=w,type='cairo-png')
par(mar=c(5,4.2,2,2))
par(xaxs='i')
par(yaxs='i')
# Example with confidence intervals and grid
mybarcol <- "gray20"
m.name  <- c('1. DLL-Kernel','2. DLL-cuBLAS', '3. PyCUDA', '4. Numba','5. TensorFlow', '6. PyTorch' )
plot(1:10, type='n',xlim=c(0,36), ylim=yl, axes=F,log='y',xlab='',ylab='',asp=1)
abline(h=yat, lty=3,col=gray(0.2))
par(new=T)
mp <- barplot2(fig1, beside = TRUE, #angle = c(-30,30,45,-45,60),density = 30,
               col = cand_col,xlim=c(0,36), ylim = yl,width=1,log="y",
               xlab='Dimension',ylab='Computation Time (sec.)',#names.arg=c('1','2','3','4','5'),
               main = "", font.main = 4,#c(0.001,0.1,10,1000),
               cex.names = 1, plot.ci = F,cex.lab=1.5, #grid.inc=10,
               plot.grid =F,axes=F)

mtext(side = 1, at =as.vector(mp), line = 0,
      text = rep(1:6,5), col = "black",cex=0.9)
mtext(side = 1, at = colMeans(mp), line = 1.5,
      text = cand_n, col = "black",cex=1.2)
legend('topleft',m.name,fill= cand_col, cex=1.2)#,angle = c(-30,30,45,-45,60),density = 30)
box()
axis(2, at=yat,labels=ylb, col.axis="black", las=1)
dev.off()



fig2 <- t(d_mean_mat)
yl <- c(1e-3,1e0)
yat <- 10^(-3:0)
ylb <- c(expression(10^-3),expression(10^-2),expression(10^-1),expression(10^0))

w = 640
png(file="mm_example_double.png",width=w, height=w,type='cairo-png')
par(mar=c(5,4.2,2,2))
par(xaxs='i')
par(yaxs='i')
# Example with confidence intervals and grid
mybarcol <- "gray20"
m.name  <- c('1. DLL-Kernel','2. DLL-cuBLAS', '3. PyCUDA', '4. Numba','5. TensorFlow', '6. PyTorch' )
plot(1:10, type='n',xlim=c(0,36), ylim=yl, axes=F,log='y',xlab='',ylab='',asp=1)
abline(h=yat, lty=3,col=gray(0.2))
par(new=T)
mp <- barplot2(fig2, beside = TRUE, #angle = c(-30,30,45,-45,60),density = 30,
               col = cand_col,xlim=c(0,36), ylim = yl,width=1,log="y",
               xlab='Dimension',ylab='Computation Time (sec.)',#names.arg=c('1','2','3','4','5'),
               main = "", font.main = 4,#c(0.001,0.1,10,1000),
               cex.names = 1, plot.ci = F,cex.lab=1.5, #grid.inc=10,
               plot.grid =F,axes=F)

mtext(side = 1, at =as.vector(mp), line = 0,
      text = rep(1:6,5), col = "black",cex=0.9)
mtext(side = 1, at = colMeans(mp), line = 1.5,
      text = cand_n, col = "black",cex=1.2)
legend('topleft',m.name,fill= cand_col, cex=1.2)#,angle = c(-30,30,45,-45,60),density = 30)
box()
axis(2, at=yat,labels=ylb, col.axis="black", las=1)
dev.off()


d_mean_mat - s_mean_mat
