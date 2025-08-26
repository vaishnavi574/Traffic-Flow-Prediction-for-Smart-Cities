import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.evaluation.RegressionEvaluator

object TrafficFlowPrediction {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("Traffic Flow Prediction")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Load CSV dataset
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("src/main/resources/traffic_data.csv")

    // Clean and transform data
    val cleanedDF = df.withColumn("timestamp", to_timestamp(col("timestamp"), "yyyy-MM-dd HH:mm:ss"))
      .withColumn("vehicle_count", col("vehicle_count").cast("int"))
      .withColumn("avg_speed", col("avg_speed").cast("double"))

    // Feature engineering
    val featureDF = cleanedDF.withColumn("hour", hour(col("timestamp")))

    val assembler = new VectorAssembler()
      .setInputCols(Array("hour", "avg_speed"))
      .setOutputCol("features")

    val finalDF = assembler.transform(featureDF).select("features", "vehicle_count")

    // Split data into training and test sets
    val Array(train, test) = finalDF.randomSplit(Array(0.8, 0.2))

    // Train Linear Regression model
    val lr = new LinearRegression()
      .setLabelCol("vehicle_count")
      .setFeaturesCol("features")

    val model = lr.fit(train)

    // Make predictions
    val predictions = model.transform(test)

    // Evaluate model
    val evaluator = new RegressionEvaluator()
      .setLabelCol("vehicle_count")
      .setPredictionCol("prediction")
      .setMetricName("rmse")

    val rmse = evaluator.evaluate(predictions)
    println(s"Root Mean Squared Error (RMSE): $rmse")

    // Optional: Show predictions
    predictions.select("features", "vehicle_count", "prediction").show(25)

    spark.stop()
  }
}

