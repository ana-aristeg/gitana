#04:00
#Optimizacion Bayesiana de hiperparametros de  rpart
#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("rlist")
require("yaml")

require("rpart")
require("parallel")

#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")


#para poder usarlo en la PC y en la nube
switch ( Sys.info()[['sysname']],
         Windows = { directory.root   <-  "M:\\" },   #Microsoft Windows
         Darwin  = { directory.root   <-  "~/dm/" },  #Apple MAC
         Linux   = { directory.root   <-  "~/buckets/b1/crudo/" }  #Entorno Google Cloud
)
#defino la carpeta donde trabajo
setwd( directory.root )


kfinalize  <- FALSE
kexperimento  <- NA   #NA si se corre la primera vez, un valor concreto si es para continuar procesando

kscript           <- "225_rpart_BO"
karch_generacion  <- "./datasetsOri/paquete_premium_202011.csv"
karch_aplicacion  <- "./datasetsOri/paquete_premium_202101.csv"
kBO_iter    <-  100   #cantidad de iteraciones de la Optimizacion Bayesiana

hs <- makeParamSet(
  makeNumericParam("cp"       , lower= -0.5  , upper=   0.1),
  makeIntegerParam("minsplit" , lower= 5L    , upper= 900L),  #la letra L al final significa ENTERO
  makeIntegerParam("minbucket", lower= 1L    , upper= 500L),
  makeIntegerParam("maxdepth" , lower= 3L    , upper=  25L),
  forbidden = quote( minbucket > 0.5*minsplit )
)


ksemilla_azar  <-  c(102191, 200177, 410551, 552581, 892237)  #Aqui poner las propias semillas
#------------------------------------------------------------------------------

get_experimento  <- function()
{
  if( !file.exists( "./maestro.yaml" ) )  cat( file="./maestro.yaml", "experimento: 1000" )
  
  exp  <- read_yaml( "./maestro.yaml" )
  experimento_actual  <- exp$experimento
  
  exp$experimento  <- as.integer(exp$experimento + 1)
  Sys.chmod( "./maestro.yaml", mode = "0644", use_umask = TRUE)
  write_yaml( exp, "./maestro.yaml" )
  Sys.chmod( "./maestro.yaml", mode = "0444", use_umask = TRUE) #dejo el archivo readonly
  
  return( experimento_actual )
}
#------------------------------------------------------------------------------

if( is.na(kexperimento ) )   kexperimento <- get_experimento()  #creo el experimento

#en estos archivos queda el resultado
kbayesiana  <- paste0("./work/E",  kexperimento, "_rpart.RDATA" )
klog        <- paste0("./work/E",  kexperimento, "_rpart_log.txt" )
kimp        <- paste0("./work/E",  kexperimento, "_rpart_importance.txt" )
kmbo        <- paste0("./work/E",  kexperimento, "_rpart_mbo.txt" )
kmejor      <- paste0("./work/E",  kexperimento, "_rpart_mejor.yaml" )
kkaggle     <- paste0("./kaggle/E",kexperimento, "_rpart_kaggle_" )

#------------------------------------------------------------------------------

loguear  <- function( reg, pscript, parch_generacion, arch)
{
  if( !file.exists(  arch ) )
  {
    linea  <- paste0( "script\tdataset\tfecha\t", 
                      paste( list.names(reg), collapse="\t" ), "\n" )
    
    cat( linea, file=arch )
  }
  
  linea  <- paste0( pscript, "\t",
                    parch_generacion, "\t",
                    format(Sys.time(), "%Y%m%d %H%M%S"),
                    "\t",
                    gsub( ", ", "\t", toString( reg ) ),
                    "\n" )
  
  cat( linea, file=arch, append=TRUE )
}
#------------------------------------------------------------------------------
#funcion que va a optimizar la Bayesian Optimization

EstimarGanancia <- function( psemilla, x, pdataset )
{
  gc()
  #divido en training/testing
  set.seed( psemilla )
  
  
  #divido en forma ESTRATIFICADA  en training/testing
  trainpct  <- 0.7   #porcentaje que se usa para training
  
  fold  <- c()
  clases  <- unique( pdataset$clase_ternaria )
  for( clase in clases )
  {
    qty  <- nrow( pdataset[ clase_ternaria==clase ] )
    vnuevo  <-  rep( 1, qty )
    vnuevo[ 1:(qty*trainpct) ] <- 0
    vnuevo  <- sample( vnuevo )
    fold  <- c( fold, vnuevo )
  }
  
  #genero modelo sobre training
  modelo  <- rpart("clase_ternaria ~ . - numero_de_cliente  - mcuentas_saldo",
                   data= pdataset[ fold==0, ],
                   xval= 0, 
                   cp=        x$cp, 
                   minsplit=  x$minsplit,
                   minbucket= x$minbucket,
                   maxdepth=  x$maxdepth
  )
  
  #aplico el modelo a los datos de testing, fold==1
  prediccion  <- predict( modelo, pdataset[ fold==1, ], type = "prob")
  
  pdataset[ fold==1, estimulo := (prediccion[ , "BAJA+2"] > 0.025) ]
  
  ganancia_testing  <- sum(  pdataset[ fold==1 & estimulo==TRUE,  ifelse( clase_ternaria=="BAJA+2", 48750, -1250 ) ] )
  ganancia_testing_normalizada  <- ganancia_testing / (1-trainpct)
  
  pdataset[ , estimulo:= NULL ]
  pdataset[ , fold:= NULL ]
  
  return( ganancia_testing_normalizada )
}
#------------------------------------------------------------------------------
#Atencion con esta funcion que SOLO puede recibir un parametro

EstimarGananciaMontecarlo <- function( x )
{
  gc()
  ganancias  <- mcmapply( EstimarGanancia, 
                          ksemilla_azar, 
                          MoreArgs= list( x, dataset), 
                          SIMPLIFY= FALSE,
                          mc.cores= 1
  )
  
  ganancia_promedio  <- mean( unlist(ganancias), na.rm=TRUE )
  
  xx  <- x
  xx$ganancia  <- ganancia_promedio
  loguear( xx, kscript, karch_generacion, klog )
  
  return( ganancia_promedio )
}
#------------------------------------------------------------------------------
#Aqui empieza el programa

#cargo el dataset
dataset  <- fread(karch_generacion)
setorder( dataset, clase_ternaria )


#Aqui comienza la configuracion de la Bayesian Optimization

configureMlr( show.learner.output = FALSE)

funcion_optimizar  <-  EstimarGananciaMontecarlo

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
#por favor, no desesperarse por lo complejo
obj.fun  <- makeSingleObjectiveFunction(
  fn=       funcion_optimizar,
  minimize= FALSE,   #estoy Maximizando la ganancia
  noisy=    TRUE,
  par.set=  hs,
  has.simple.signature = FALSE
)


ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  save.file.path= kbayesiana)
ctrl  <- setMBOControlTermination(ctrl, iters= kBO_iter )
ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI())

surr.km  <-  makeLearner("regr.km", predict.type= "se", covtype= "matern3_2", control= list(trace = TRUE))

if( kfinalize )
{
  mboFinalize(kbayesiana)
}

if(!file.exists(kbayesiana))
{
  #lanzo la busqueda bayesiana
  run  <- mbo(obj.fun, learner = surr.km, control = ctrl)
} else {
  
  #retoma el procesamiento en donde lo dejo
  run  <- mboContinue( kbayesiana ) 
}


#ordeno las corridas
tbl  <- as.data.table(run$opt.path)
tbl[ , iteracion := .I ]  #le pego el numero de iteracion
setorder( tbl, -y )

#agrego info que me viene bien
tbl[ , script          := kscript ]
tbl[ , arch_generacion := karch_generacion ]
tbl[ , arch_aplicacion := karch_aplicacion ]

fwrite(  tbl, file=kmbo, sep="\t" )   #grabo TODA la corrida
write_yaml( tbl[1], kmejor )          #grabo el mejor

#------------------------------------------------------------------------------
#genero las mejores 5  salidas para Kaggle

#cargo los datos de 201907, que es donde voy a APLICAR el modelo
dapply  <- fread(karch_aplicacion)

for( modelito in 1:5 )
{
  x  <- tbl[modelito]   #en x quedaron los MEJORES hiperparametros
  
  modelo  <- rpart("clase_ternaria ~ . - numero_de_cliente  - mcuentas_saldo",
                   data= dataset,
                   xval= 0, 
                   cp=        x$cp, 
                   minsplit=  x$minsplit,
                   minbucket= x$minbucket,
                   maxdepth=  x$maxdepth
  )
  
  #importancia de variables
  tb_importancia <-  as.data.table( list( "Feature"= names(modelo$variable.importance), 
                                          "Importance"= modelo$variable.importance )  )
  fwrite( tb_importancia, kimp, sep="\t" )
  
  
  #genero el vector con la prediccion, la probabilidad de ser positivo
  prediccion  <- predict( modelo, dapply)
  
  dapply[ , prob_baja2 := prediccion[, "BAJA+2"] ]
  dapply[ , estimulo   := ifelse( prob_baja2 > 0.025, 1, 0 ) ]
  
  entrega  <-  dapply[   , list( numero_de_cliente, estimulo)  ]
  
  #genero el archivo para Kaggle
  fwrite( entrega, 
          file= paste0(kkaggle, modelito, ".csv" ),
          sep=  "," )
  
}

