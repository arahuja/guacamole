package org.hammerlab.guacamole.commands

import breeze.linalg.DenseVector
import org.apache.spark.SparkContext
import org.apache.spark.ml.classification.{BinaryLogisticRegressionSummary, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.StandardScaler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.linalg.{DenseVector => SparkDenseVector}
import org.apache.spark.sql.functions._
import org.hammerlab.guacamole.distributed.PileupFlatMapUtils.pileupFlatMapTwoSamples
import org.hammerlab.guacamole.loci.parsing.ParsedLoci
import org.hammerlab.guacamole.loci.set.LociSet
import org.hammerlab.guacamole.pileup.Pileup
import org.hammerlab.guacamole.readsets.ReadSets
import org.hammerlab.guacamole.readsets.args.{ReferenceArgs, TumorNormalReadsArgs}
import org.hammerlab.guacamole.readsets.rdd.{PartitionedRegions, PartitionedRegionsArgs}
import org.hammerlab.guacamole.reference.{ContigName, Locus, ReferenceBroadcast}
import org.hammerlab.guacamole.variants.{Allele, AlleleEvidence, Genotype}
import org.kohsuke.args4j.{Option => Args4jOption}

  class SomaticFilterModelArgs
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

    @Args4jOption(name = "--model-output", required = true, usage = "")
    var modelOutput: String = ""

  }

  object SomaticFilterModel extends SparkCommand[SomaticFilterModelArgs] {
    override val name = "somatic-filter-model"
    override val description = ""

    override def run(args: SomaticFilterModelArgs, sc: SparkContext) = {

      val sqlContext = new org.apache.spark.sql.SQLContext(sc)

      // this is used to implicitly convert an RDD to a DataFrame.
      import sqlContext.implicits._

      val reference = args.reference(sc)

      val (readsets, loci) = ReadSets(sc, args)

      val trueLociSet =
        ParsedLoci
          .loadFromFile(args.trueLoci, sc.hadoopConfiguration)
          .result(readsets.contigLengths)

      val positiveLoci =
        computeLociEvidence(sc, args, reference, readsets, trueLociSet)
          .map(v => new SparkDenseVector(v.data))
          .keyBy(x => 1.0)


      val falseLoci =
        if (args.falseLoci.nonEmpty)
          ParsedLoci
            .loadFromFile(args.falseLoci, sc.hadoopConfiguration)
            .result(readsets.contigLengths)
        else
          ParsedLoci(trueLociSet.contigs.map(_.name).mkString(","))
            .result(readsets.contigLengths)
            .difference(trueLociSet)

      val negativeLoci =
        computeLociEvidence(sc, args, reference, readsets, falseLoci)
          .map(v => new SparkDenseVector(v.data))
          .keyBy(x => 0.0)

      val dataset = (positiveLoci ++ negativeLoci).toDF("label", "unscaled_features")

      val scaler = new StandardScaler()
        .setWithStd(true)
        .setWithMean(true)
        .setInputCol("unscaled_features")
        .setOutputCol("features")

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
        .setNumFolds(4)


      val cvModel = cv.fit(dataset)
      // val cvModel = CrossValidatorModel.load("filter.model")//cv.fit(dataset)

      val bestp =   cvModel.bestModel.asInstanceOf[PipelineModel]
      val fitlr = bestp.stages(1).asInstanceOf[LogisticRegressionModel]
      println (fitlr.coefficients.toDense)

      cvModel.avgMetrics.map(println)
      cvModel.write.overwrite().save(args.modelOutput)


      // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
      // example
      val binarySummary = fitlr.summary.asInstanceOf[BinaryLogisticRegressionSummary]

      // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
      val roc = binarySummary.roc
      roc.show()
      println(binarySummary.areaUnderROC)

      // Set the model threshold to maximize F-Measure
      val fMeasure = binarySummary.fMeasureByThreshold
      val maxFMeasure = fMeasure.select(max("F-Measure")).head().getDouble(0)
      val bestThreshold = fMeasure.where($"F-Measure" === maxFMeasure)
        .select("threshold").head().getDouble(0)

      println(bestThreshold)


    }

    def computeLociEvidence(sc: SparkContext,
                            args: SomaticFilterModelArgs,
                            reference: ReferenceBroadcast,
                            readsets: ReadSets,
                            lociSet: LociSet) = {

      val partitionedReads =
        PartitionedRegions(
          readsets.allMappedReads,
          lociSet,
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
          if (pileupTumor.referenceDepth != pileupTumor.depth)
            computePileupStats(
              pileupTumor,
              pileupNormal
            )._1.iterator
          else
            Iterator.empty,
        reference = reference
      )
    }

    def computePileupStats(tumorPileup: Pileup,
                           normalPileup: Pileup): (Option[DenseVector[Double]], ContigName, Locus, Allele) = {

      val referenceAllele = Allele(tumorPileup.referenceBase, tumorPileup.referenceBase)

      val contigName = tumorPileup.contigName
      val locus = tumorPileup.locus

      // For now, we skip loci that have no reads mapped. We may instead want to emit NoCall in this case.
      if (tumorPileup.elements.isEmpty
        || normalPileup.elements.isEmpty
        || tumorPileup.referenceDepth == tumorPileup.depth // skip computation if no alternate reads
      )
        return (None, contigName, locus, referenceAllele)

      val tumorDepth = tumorPileup.depth
      val variantAlleleFractions: Map[Allele, Double] =
        tumorPileup
          .elements
          .withFilter(_.allele.isVariant)
          .map(_.allele)
          .groupBy(identity)
          .map(kv => kv._1 -> kv._2.size / tumorDepth.toDouble )


      val referenceGenotype = Genotype(Map(referenceAllele -> 1.0))

      val mostFrequentVariantAllele = variantAlleleFractions.maxBy(_._2)
      val variantAllele =  mostFrequentVariantAllele._1
      val empiricalVariantAlleleFrequency =  mostFrequentVariantAllele._2

      val evidences =
        for {
          pileup <- Vector(tumorPileup, normalPileup)
          allele <- Vector(referenceAllele, variantAllele)
        }
          yield AlleleEvidence(1, allele, pileup).toDenseVector

      (Some(DenseVector.vertcat(evidences:_*)), contigName, locus, variantAllele)
  }
}
