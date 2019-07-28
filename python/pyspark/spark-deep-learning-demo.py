from pyspark.ml.image import ImageSchema
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.sql.functions import lit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


tulips_path = "/Users/ferro/git/ohtar/wtsp/datasets/flower_photos/tulips"
daisy_path = "/Users/ferro/git/ohtar/wtsp/datasets/flower_photos/daisy"

tulips_df = ImageSchema.readImages(tulips_path).withColumn("label", lit(1))
daisy_df = ImageSchema.readImages(daisy_path).withColumn("label", lit(2))

tulips_train, tulips_test, _ = tulips_df.randomSplit([0.1, 0.05, 0.85])
daisy_train, daisy_test, _ = daisy_df.randomSplit([0.1, 0.05, 0.85])

train_df = tulips_train.unionAll(daisy_train)
test_df = tulips_test.unionAll(daisy_test)

# repartition to only have 100 samples in memory to avoid overhead
train_df = train_df.repartition(100)
test_df = test_df.repartition(100)

featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])

model = p.fit(train_df)

df = model.transform(image_df.limit(10)).select("image", "probability", "uri", "label")
predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
accuracy = evaluator.evaluate(predictionAndLabels)

print( "Training set accuracy {}".format(accuracy) )

"""
pyspark --driver-memory 8G --packages databricks:spark-deep-learning:1.5.0-spark2.4-s_2.11,com.twelvemonkeys.imageio:imageio-jpeg:3.3.1
"""