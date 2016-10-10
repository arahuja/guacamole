package org.hammerlab.guacamole.commands

import breeze.linalg.DenseVector
import org.apache.spark.SparkContext
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.linalg.{DenseVector => SparkDenseVector}
import org.hammerlab.guacamole.distributed.PileupFlatMapUtils.pileupFlatMapTwoSamples
import org.hammerlab.guacamole.loci.parsing.ParsedLoci
import org.hammerlab.guacamole.pileup.Pileup
import org.hammerlab.guacamole.readsets.ReadSets
import org.hammerlab.guacamole.readsets.args.{ReferenceArgs, TumorNormalReadsArgs}
import org.hammerlab.guacamole.readsets.rdd.{PartitionedRegions, PartitionedRegionsArgs}
import org.hammerlab.guacamole.reference.ReferenceBroadcast
import org.hammerlab.guacamole.variants.{Allele, AlleleEvidence, Genotype}
import org.kohsuke.args4j.{Option => Args4jOption}

  class Arguments
    extends Args
      with TumorNormalReadsArgs
      with PartitionedRegionsArgs
      with ReferenceArgs {

    @Args4jOption(name = "--dbsnp-vcf", required = false, usage = "VCF file to identify DBSNP variants")
    var dbSnpVcf: String = ""

    @Args4jOption(name = "--true-loci", required = true, usage = "")
    var trueLoci: String = ""

    @Args4jOption(name = "--false-loci", required = true, usage = "")
    var falseLoci: String = ""

  }

  object SomaticFilterModel extends SparkCommand[Arguments] {
    override val name = "somatic-filter-model"
    override val description = ""

    override def run(args: Arguments, sc: SparkContext) = {

      val sqlContext = new org.apache.spark.sql.SQLContext(sc)

      // this is used to implicitly convert an RDD to a DataFrame.
      import sqlContext.implicits._

      val reference = args.reference(sc)

      val (readsets, loci) = ReadSets(sc, args)

      val positiveLoci =
        computeLociEvidence(sc, args, reference, readsets, args.trueLoci)
          .map(v => new SparkDenseVector(v.data))
          .keyBy(x => 1.0)

      val negativeLoci =
        computeLociEvidence(sc, args, reference, readsets, args.falseLoci)
          .map(v => new SparkDenseVector(v.data))
          .keyBy(x => 0.0)

      val dataset = (positiveLoci ++ negativeLoci).toDF("label", "unscaled_features")

      val scaler = new StandardScaler()
        .setWithStd(true)
        .setWithMean(true)
        .setInputCol("unscaled_features")
        .setOutputCol("features")

      (positiveLoci ++ negativeLoci).map(_._2.toString).saveAsTextFile("/Users/arahuja/test.out")

      val lr = new LogisticRegression()
        .setMaxIter(10)

      val pipeline = new Pipeline()
        .setStages(Array(scaler, lr))


      val paramGrid = new ParamGridBuilder()
        .addGrid(lr.regParam, Array(0.1, 0.01))
        .build()

      val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(new BinaryClassificationEvaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(2)

      val cvModel = cv.fit(dataset)

      cvModel.avgMetrics.map(println)
    }

    def computeLociEvidence(sc: SparkContext,
                            args: Arguments,
                            reference: ReferenceBroadcast,
                            readsets: ReadSets,
                            lociFile: String) = {
      val trueLociSet =
        ParsedLoci
          .loadFromFile(lociFile, sc.hadoopConfiguration)
          .result(readsets.contigLengths)

      val partitionedReads =
        PartitionedRegions(
          readsets.allMappedReads,
          trueLociSet,
          args
        )

      val normalSampleName = args.normalSampleName
      val tumorSampleName = args.tumorSampleName

      pileupFlatMapTwoSamples[DenseVector[Double]](
        partitionedReads,
        sample1Name = normalSampleName,
        sample2Name = tumorSampleName,
        skipEmpty = true, // skip empty pileups
        function = (pileupNormal, pileupTumor) =>
          computePileupStats(
            pileupTumor,
            pileupNormal
          ).iterator,
        reference = reference
      )
    }

    def computePileupStats(tumorPileup: Pileup,
                           normalPileup: Pileup): Option[DenseVector[Double]] = {

      val tumorDepth = tumorPileup.depth
      val variantAlleleFractions: Map[Allele, Double] =
        tumorPileup
          .elements
          .withFilter(_.allele.isVariant)
          .map(_.allele)
          .groupBy(identity)
          .map(kv => kv._1 -> kv._2.size / tumorDepth.toDouble )


      val referenceAllele = Allele(tumorPileup.referenceBase, tumorPileup.referenceBase)
      val referenceGenotype = Genotype(Map(referenceAllele -> 1.0))

      val mostFrequentVariantAllele = variantAlleleFractions.maxBy(_._2)
      val variantAllele =  mostFrequentVariantAllele._1
      val empiricalVariantAlleleFrequency =  mostFrequentVariantAllele._2

      val evidences =
        for {
          pileup <- Vector(tumorPileup, normalPileup)
          allele <- Vector(referenceAllele)
        }
          yield AlleleEvidence(1, allele, pileup).toDenseVector

      Some(DenseVector.vertcat(evidences:_*))

  }
}
