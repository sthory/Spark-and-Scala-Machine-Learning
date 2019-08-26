import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors

object LinearRegressionDataFrame {
  
  var diferencia:Double = 0.0
  var acum_error:Double = 0.00
  var i:Double = 0.0
  /** main function */
  def main(args: Array[String]) {
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    
    // Use new SparkSession interface in Spark 2.0
    val spark = SparkSession
      .builder
      .appName("LinearRegressionDF")
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "file:///C:/temp")
      .getOrCreate()
    
    val inputLines = spark.sparkContext.textFile("../regression.txt")
    val data = inputLines.map(_.split(",")).map(x => (x(0).toDouble, Vectors.dense(x(1).toDouble)))
    
    // Convert this RDD to a DataFrame
    import spark.implicits._
    val colNames = Seq("label", "features")
    val df = data.toDF(colNames: _*)
    
 
    // Let's split our data into training data and testing data
    val trainTest = df.randomSplit(Array(0.5, 0.5))
    val trainingDF = trainTest(0)
    val testDF = trainTest(1)
    
    // Now create our linear regression model
    val lir = new LinearRegression()
      .setRegParam(0.3) // regularization 
      .setElasticNetParam(0.8) // elastic net mixing
      .setMaxIter(100) // max iterations
      .setTol(1E-6) // convergence tolerance
    
    // Train the model using our training data
    val model = lir.fit(trainingDF)
    
    // Now see if we can predict values in our test data.
    val fullPredictions = model.transform(testDF).cache()
    
    // Extract the predictions and the "known" correct labels.
    val predictionAndLabel = fullPredictions.select("prediction", "label").rdd.map(x => (x.getDouble(0), x.getDouble(1)))
    
    // Print out the predicted and actual values for each point
    for (prediction <- predictionAndLabel) {
      println(prediction)
      diferencia = prediction._1 - prediction._2
      diferencia = diferencia * diferencia
      acum_error += diferencia
      i += 1
    }
    
    var error_total = 0.0
    error_total = acum_error / i 
    println("El error total es: " + error_total)
    
    // Stop the session
    spark.stop()

  }
}