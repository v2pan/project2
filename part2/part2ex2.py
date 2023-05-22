import json
import pandas as pd
from pyspark import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer, StopWordsRemover, ChiSqSelector, IDF
from pyspark.sql import SparkSession

file = open("./reviews_devset.json")

reviews_devset = file.readlines()

reviewsDF = pd.DataFrame.from_records(list(map(json.loads,reviews_devset)))

print(reviewsDF)

reviewsDF["category"] = pd.Categorical(reviewsDF["category"])
reviewsDF["labels"] = reviewsDF["category"].cat.codes
print(reviewsDF.loc[:,["reviewText","category","labels"]])

spark = SparkSession.builder.master("local[1]").appName("part2ex2").getOrCreate()

spark_dff = spark.createDataFrame(reviewsDF.loc[0:1000,["reviewText","category","labels"]])

# reviewsRDD = sc.parallelize(spark_dff)

tokenizer = Tokenizer(inputCol="reviewText", outputCol="rawWords")

remover = StopWordsRemover(
    inputCol=tokenizer.getOutputCol(),
    outputCol="words",
    caseSensitive=False)

hashingTF = HashingTF(
    inputCol=remover.getOutputCol(), outputCol="rawFeatures", numFeatures=2000)

idf = IDF(inputCol=hashingTF.getOutputCol(), outputCol="features")

selector = ChiSqSelector(
    numTopFeatures=2000, 
    featuresCol=idf.getOutputCol(),
    outputCol="selectedFeatures", 
    labelCol="labels"
    )

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, selector])

model = pipeline.fit(spark_dff)

out = model.transform(spark_dff)

out.toPandas().to_csv('test.csv')
