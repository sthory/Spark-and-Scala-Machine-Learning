import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

// Read DATA
val data = (spark.read.option("header","true")
            .option("inferSchema","true")
            .option("multiline", "true")
            .format("csv")
            .load("../Regression/USA_Housing.csv"))

data.printSchema

// Rename label column
// Grab only numerical columns
val df = (data.select(data("Price").as("label"),
                      $"Avg Area Income",
                      $"Avg Area House Age",
                      $"Avg Area Number of Rooms",
                      $"Area Population"))

////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Set the input columns from which we are supposed to read the values
// Set the name of the column where the vector will be stored
val assembler = (new VectorAssembler()
                .setInputCols(Array("Avg Area Income",
                                    "Avg Area House Age",
                                    "Avg Area Number of Rooms",
                                    "Area Population"))
                .setOutputCol("features"))

// Transform the DataFrame
val output = assembler.transform(df).select($"label",$"features")

// Training and Test DATA
val Array(training, test) = (output.select($"label",$"features")
                            .randomSplit(Array(0.7,0.3),seed=12345))

// model
val lr = new LinearRegression()

/////////////////////////////////////
/// PARAMETER GRID BUILDER //////////
////////////////////////////////////
val paramGrid = new ParamGridBuilder().addGrid(lr.regParam,Array(10000000,0.0001)).build()

// Train SPLIT (Holdout)
val trainValidationSplit = (new TrainValidationSplit()
                            .setEstimator(lr)
                            .setEvaluator(new RegressionEvaluator()
                                          .setMetricName("r2"))
                            .setEstimatorParamMaps(paramGrid)
                            .setTrainRatio(0.8))

val model = trainValidationSplit.fit(training)

model.transform(test).select("features","label","prediction").show()

// El primer resultado da un modelo muy malo (0.0616438), el segundo es
// muy bueno (0.9278)
model.validationMetrics
