# This code is from Jack in Facebook Checkin Competition on Kaggle
library(data.table)
library(dplyr) 
library(xgboost)

system("ls ../input")

data.train <- fread("../input/train.csv", colClasses=list(character="place_id"))
data.test <- fread("../input/test.csv")

train.end <- max(data.train$time)
test.end <- max(data.test$time)


############################## step 1 ##############################

cat("step: 1/5\n")

num.x1 <- 100
num.y1 <- 200

temp <- data.train[, .(place_id, new.x1 = as.integer(floor(x/(10+1E-10)*num.x1)), new.y1 = as.integer(floor(y/(10+1E-10)*num.y1)))]
temp <- temp[, .(count = .N), by=c("new.x1","new.y1","place_id")]

temp <- arrange(temp, new.x1, new.y1, desc(count))
temp$rank <- ifelse(duplicated(paste(temp$new.x1, temp$new.y1)), 0, 1)

temp <- subset(temp, count >= 2)
n.candidate <- max(temp[ , list(count=.N), by=c("new.x1","new.y1")]$count)

i <- 2
while(sum(temp$rank==0)) {
        temp$rank <- ifelse(temp$rank == 0 & c(0,temp$rank[-nrow(temp)]) == i-1, i, temp$rank)
        i <- i+1
}

candidate <- subset(temp, rank==1)[, .(new.x1, new.y1, place_id)]
for(i in 2:n.candidate) {
        candidate <- merge(candidate, subset(temp, rank==i)[, 1:3, with=FALSE], by=c("new.x1","new.y1"), all.x=TRUE, suffixes=c("",paste0("_",i)))
}
colnames(candidate)[3:ncol(candidate)] <- paste0("candidate_", 1:n.candidate)
write.csv(candidate, "candidate.csv", row.names=FALSE)

rm(temp)

############################## step 2 ##############################

cat("step: 2/5\n")

predict.interval <- 24*60*7*2

freq <- data.table(place_id=unique(data.train$place_id))
for(i in 1:(train.end/predict.interval)) {
        temp <- data.train[train.end-predict.interval*i < time & time <= train.end-predict.interval*(i-1), .N, by="place_id"]
        freq <- merge(freq, temp, by="place_id", all.x=TRUE)
        colnames(freq)[ncol(freq)] <- paste0("freq_", as.integer(train.end-predict.interval*i), "-", as.integer(train.end-predict.interval*(i-1)))
}
freq[is.na(freq)] <- 0

params <- list("eta"=0.1, "max_depth"=6, "min_child_weight"=100, "objective"="reg:linear", "eval_metric"="rmse")

pr.place.table <- freq[, .(place_id)]
for(i in 1:ceiling((test.end-train.end)/predict.interval)) {
        
        cat(paste0("    substep: ", i, "/", ceiling((test.end-train.end)/predict.interval),"\n"))
        
        x.train <- freq[, (i+2):ncol(freq), with=FALSE]
        y.train <- freq[[2]][apply(x.train, 1, sum) > 0]
        x.train <- x.train[apply(x.train, 1, sum) > 0]
        
        x.test <- freq[, 2:(ncol(freq)-i), with=FALSE]
        
        x.train <- as.matrix(x.train)*1.0
        x.test <- as.matrix(x.test)*1.0
        
        if(i == 1) {
                nrounds <- 100
        } else if(i == 2) {
                nrounds <- 30
        } else {
                nrounds <- 10
        }
        
        set.seed(0)
        model.xgb <- xgb.train(param=params, data=xgb.DMatrix(x.train, label=y.train), nrounds=nrounds)
        
        temp <- predict(model.xgb, x.test)
        temp[temp<0] <- 0
        temp <- temp/sum(temp)
        pr.place.table <- cbind(pr.place.table, temp)
        colnames(pr.place.table)[ncol(pr.place.table)] <- paste0("pr_", as.integer(train.end+predict.interval*(i-1)), "-", as.integer(train.end+predict.interval*i))
        
}

write.csv(pr.place.table, "pr_place.csv", row.names=FALSE)

rm(freq)
rm(x.train)
rm(x.test)
rm(temp)

############################## step 3 ##############################

cat("step: 3/5\n")

########## x distibution ##########

cat("    substep: 1/5\n")

num.x2 <- 1200

temp <- data.train[, .(place_id, new.x2 = as.integer(floor(x/(10+1E-10)*num.x2)), time)]
temp <- temp[, .(count = sum(1-0.8*(train.end-time)^0.2/train.end^0.2)), by=c("place_id","new.x2")]

dist.x <- temp
dist.x <- merge(dist.x, temp[, .(place_id, new.x2 = new.x2+1, count)], by=c("place_id","new.x2"), suffixes=c("", ".p1"), all=TRUE)
dist.x <- merge(dist.x, temp[, .(place_id, new.x2 = new.x2-1, count)], by=c("place_id","new.x2"), suffixes=c("", ".m1"), all=TRUE)
dist.x <- merge(dist.x, temp[, .(place_id, new.x2 = new.x2+2, count)], by=c("place_id","new.x2"), suffixes=c("", ".p2"), all=TRUE)
dist.x <- merge(dist.x, temp[, .(place_id, new.x2 = new.x2-2, count)], by=c("place_id","new.x2"), suffixes=c("", ".m2"), all=TRUE)

dist.x[is.na(dist.x)] <- 0
dist.x <- dist.x[, .(place_id, new.x2, count.x = (count + 0.8*(count.p1+count.m1) + 0.6*(count.p2+count.m2)) / 3.8)]

rm(temp)

########## y distribution ##########

cat("    substep: 2/5\n")

num.y2 <- 4000

temp <- data.train[, .(place_id, new.y2 = as.integer(floor(y/(10+1E-10)*num.y2)), time)]
temp <- temp[, .(count = sum(1-0.8*(train.end-time)^0.2/train.end^0.2)), by=c("place_id","new.y2")]

dist.y <- temp
dist.y <- merge(dist.y, temp[, .(place_id, new.y2 = new.y2+1, count)], by=c("place_id","new.y2"), suffixes=c("", ".p1"), all=TRUE)
dist.y <- merge(dist.y, temp[, .(place_id, new.y2 = new.y2-1, count)], by=c("place_id","new.y2"), suffixes=c("", ".m1"), all=TRUE)
dist.y <- merge(dist.y, temp[, .(place_id, new.y2 = new.y2+2, count)], by=c("place_id","new.y2"), suffixes=c("", ".p2"), all=TRUE)
dist.y <- merge(dist.y, temp[, .(place_id, new.y2 = new.y2-2, count)], by=c("place_id","new.y2"), suffixes=c("", ".m2"), all=TRUE)

dist.y[is.na(dist.y)] <- 0
dist.y <- dist.y[, .(place_id, new.y2, count.y = (count + 0.8*(count.p1+count.m1) + 0.6*(count.p2+count.m2)) / 3.8)]

rm(temp)

########## time of day distribution ##########

cat("    substep: 3/5\n")

num.d <- 40

temp <- data.train[, .(place_id, new.d = as.integer(floor((time%%(60*24))/(60*24+1E-10)*num.d)))]
temp <- temp[, .(count = .N), by=c("place_id","new.d")]

dist.day <- temp
dist.day <- merge(dist.day, temp[, .(place_id, new.d = as.integer((new.d+1)%%num.d)      , count)], by=c("place_id","new.d"), suffixes=c("", ".p1"), all=TRUE)
dist.day <- merge(dist.day, temp[, .(place_id, new.d = as.integer((new.d+num.d-1)%%num.d), count)], by=c("place_id","new.d"), suffixes=c("", ".m1"), all=TRUE)
dist.day <- merge(dist.day, temp[, .(place_id, new.d = as.integer((new.d+2)%%num.d)      , count)], by=c("place_id","new.d"), suffixes=c("", ".p2"), all=TRUE)
dist.day <- merge(dist.day, temp[, .(place_id, new.d = as.integer((new.d+num.d-2)%%num.d), count)], by=c("place_id","new.d"), suffixes=c("", ".m2"), all=TRUE)

dist.day[is.na(dist.day)] <- 0
dist.day <- dist.day[, .(place_id, new.d, count.d = (count + 0.8*(count.p1+count.m1) + 0.6*(count.p2+count.m2)) / 3.8)]

rm(temp)

########## day of week distribution ##########

cat("    substep: 4/5\n")

num.w <- 7*3
temp <- data.train[, .(place_id, new.w = as.integer(floor((time%%(60*24*7))/(60*24*7+1E-10)*num.w)))]
temp <- temp[, .(count = .N), by=c("place_id","new.w")]

dist.week <- temp
dist.week <- merge(dist.week, temp[, .(place_id, new.w = as.integer((new.w+1)%%num.w)      , count)], by=c("place_id","new.w"), suffixes=c("", ".p1"), all=TRUE)
dist.week <- merge(dist.week, temp[, .(place_id, new.w = as.integer((new.w+num.w-1)%%num.w), count)], by=c("place_id","new.w"), suffixes=c("", ".m1"), all=TRUE)

dist.week[is.na(dist.week)] <- 0
dist.week <- dist.week[, .(place_id, new.w, count.w = (count+count.p1+count.m1) / 3)]

rm(temp)

########## accuracy distribution ##########

cat("    substep: 5/5\n")

num.a <- 70

temp <- data.train[, .(place_id, new.a = as.integer(floor(log10(accuracy)/3*num.a)), time)]
temp$new.a[temp$new.a>=num.a] <- num.a-1
temp <- temp[, .(count = sum(1-0.8*(train.end-time)^0.2/train.end^0.2)), by=c("place_id","new.a")]

dist.accuracy <- temp
dist.accuracy <- merge(dist.accuracy, temp[, .(place_id, new.a = new.a+1, count)], by=c("place_id","new.a"), suffixes=c("", ".p1"), all=TRUE)
dist.accuracy <- merge(dist.accuracy, temp[, .(place_id, new.a = new.a-1, count)], by=c("place_id","new.a"), suffixes=c("", ".m1"), all=TRUE)
dist.accuracy <- merge(dist.accuracy, temp[, .(place_id, new.a = new.a+2, count)], by=c("place_id","new.a"), suffixes=c("", ".p2"), all=TRUE)
dist.accuracy <- merge(dist.accuracy, temp[, .(place_id, new.a = new.a-2, count)], by=c("place_id","new.a"), suffixes=c("", ".m2"), all=TRUE)

dist.accuracy[is.na(dist.accuracy)] <- 0
dist.accuracy <- dist.accuracy[, .(place_id, new.a, count.a = (count + 0.8*(count.p1+count.m1) + 0.6*(count.p2+count.m2)) / 3.8)]

rm(temp)

############################## step 4 ##############################

cat("step: 4/5\n")

count <- data.train[, .(count = .N, count.decay = sum(1-0.8*(train.end-time)^0.2/train.end^0.2)), by="place_id"]
rm(data.train)

my.which.max <- function(x, N=1) {
        while(N > 1) {
                x[which.max(x)] <- NA
                N <- N-1
        }
        which.max(x)
}

batch.size <- 1000000
result <- data.table()

# change the range to 1:ceiling(nrow(data.test)/batch.size)
for(i in 1:1) {
        
        cat(paste0("    substep: ", i, "/", ceiling(nrow(data.test)/batch.size),"\n"))
        
        data.batch <- data.test[((i-1)*batch.size+1):min(i*batch.size,nrow(data.test)), ] %>%
                mutate(new.x1 = as.integer(floor(x/(10+1E-10)*num.x1)),
                       new.y1 = as.integer(floor(y/(10+1E-10)*num.y1)),
                       new.x2 = as.integer(floor((x/(10+1E-10)*num.x2))),
                       new.y2 = as.integer(floor((y/(10+1E-10)*num.y2))),
                       new.d  = as.integer(floor((time%%(60*24))/(60*24+1E-10)*num.d)),
                       new.w  = as.integer(floor((time%%(60*24*7))/(60*24*7+1E-10)*num.w)),
                       new.a  = as.integer(floor(log10(accuracy)/3*num.a)))
        data.batch$new.a[data.batch$new.a>=num.a] <- num.a-1
        
        data.batch <- merge(data.batch, candidate, by=c("new.x1", "new.y1")) %>% arrange(row_id)
        
        pr.table <- data.batch[, "row_id", with=FALSE]
        start.time <- proc.time()
        end.time <- start.time
        for(j in 1:(ncol(candidate)-2)) {
                temp <- data.batch[, c("row_id","time","new.x2","new.y2","new.d","new.w","new.a",paste0("candidate_",j)), with=FALSE]
                temp <- merge(temp, dist.x,        by.x=c(paste0("candidate_",j),"new.x2"), by.y=c("place_id","new.x2"))
                temp <- merge(temp, dist.y,        by.x=c(paste0("candidate_",j),"new.y2"), by.y=c("place_id","new.y2"))
                temp <- merge(temp, dist.day,      by.x=c(paste0("candidate_",j),"new.d"),  by.y=c("place_id","new.d"), all.x=TRUE)
                temp <- merge(temp, dist.week,     by.x=c(paste0("candidate_",j),"new.w"),  by.y=c("place_id","new.w"), all.x=TRUE)
                temp <- merge(temp, dist.accuracy, by.x=c(paste0("candidate_",j),"new.a"),  by.y=c("place_id","new.a"), all.x=TRUE)
                temp <- merge(temp, pr.place.table,  by.x=paste0("candidate_",j), by.y="place_id")
                temp <- merge(temp, count, by.x=paste0("candidate_",j), by.y="place_id")
                
                pr.x <- temp$count.x / temp$count.decay
                pr.y <- temp$count.y / temp$count.decay
                
                temp$count.d[is.na(temp$count.d)] <- 0
                temp$count.d <- temp$count.d + 0.1
                pr.time_of_day <- temp$count.d / (temp$count + 0.1*num.d)
                
                temp$count.w[is.na(temp$count.w)] <- 0
                temp$count.w <- temp$count.w + 1
                pr.day_of_week <- temp$count.w / (temp$count + 1*num.w)
                
                temp$count.a[is.na(temp$count.a)] <- 0
                temp$count.a <- temp$count.a + 0.1
                pr.accuracy <- temp$count.a / (temp$count.decay + 0.1*num.a)
                
                temp$pr.place <- NA
                for(k in 1:(ncol(pr.place.table)-1)) {
                        temp$pr.place <- ifelse(is.na(temp$pr.place) & (temp$time <= train.end+predict.interval*k), temp[[colnames(pr.place.table)[k+1]]], temp$pr.place)
                }
                temp$pr.place[temp$pr.place < 4E-6] <- 4E-6
                
                temp$log.pr <- log(temp$pr.place) + log(pr.x) + log(pr.y) + log(pr.time_of_day) + log(pr.day_of_week) + log(pr.accuracy)
                temp <- temp[, .(row_id, log.pr)] %>% arrange(row_id)
                pr.table <- merge(pr.table, temp, by="row_id", all.x=TRUE)
                colnames(pr.table)[ncol(pr.table)] <- paste0("log.pr_",j)
        }
        
        result.batch <- data.batch[, .(row_id)]
        
        temp <- apply(pr.table[, 2:ncol(pr.table), with=FALSE], 1, my.which.max, 1)
        for(j in 1:(ncol(candidate)-2)) {
                result.batch[temp==j, "p1"] <- data.batch[temp==j, ][[paste0("candidate_",j)]]
        }
        
        temp <- apply(pr.table[, 2:ncol(pr.table), with=FALSE], 1, my.which.max, 2)
        result.batch$p2 <- ""
        for(j in 1:(ncol(candidate)-2)) {
                result.batch[temp==j, "p2"] <- data.batch[temp==j, ][[paste0("candidate_",j)]]
        }
        
        temp <- apply(pr.table[, 2:ncol(pr.table), with=FALSE], 1, my.which.max, 3)
        result.batch$p3 <- ""
        for(j in 1:(ncol(candidate)-2)) {
                result.batch[temp==j, "p3"] <- data.batch[temp==j, ][[paste0("candidate_",j)]]
        }
        
        result <- rbind(result, result.batch)
        
}

############################## step 5 ##############################

cat("step: 5/5\n")

result$place_id <- paste(result$p1, result$p2, result$p3)
result <- merge(data.test[, .(row_id)], result[, .(row_id, place_id)], by="row_id", all.x=TRUE)
write.csv(result, "Submission_1000000rows.csv", row.names=FALSE, quote=FALSE)
