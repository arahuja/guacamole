package org.hammerlab.guacamole.commands

import breeze.linalg.DenseVector
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{DenseVector => SparkDenseVector}
import org.apache.spark.sql.SaveMode
import org.hammerlab.guacamole.distributed.PileupFlatMapUtils.pileupFlatMapTwoSamples
import org.hammerlab.guacamole.filters.genotype.GenotypeFilter.GenotypeFilterArguments
import org.hammerlab.guacamole.likelihood.Likelihood
import org.hammerlab.guacamole.loci.set.LociSet
import org.hammerlab.guacamole.pileup.Pileup
import org.hammerlab.guacamole.readsets.ReadSets
import org.hammerlab.guacamole.readsets.args.{ReferenceArgs, TumorNormalReadsArgs}
import org.hammerlab.guacamole.readsets.rdd.{PartitionedRegions, PartitionedRegionsArgs}
import org.hammerlab.guacamole.reference.{ContigName, ContigSequence, Locus, ReferenceBroadcast}
import org.hammerlab.guacamole.util.Bases
import org.hammerlab.guacamole.variants.{Allele, AlleleEvidence, Genotype}
import org.kohsuke.args4j.{Option => Args4jOption}

case class ReferenceContext(leftContext: String,
                            rightContext: String,
                            baseCounts: Map[Byte, Int])

object ReferenceContext {
  def apply(contigSequence: ContigSequence,
            locus: Locus,
            flankingContextLength: Int,
            halfBaseContextWindow: Int): ReferenceContext = {

    val contigWindow = contigSequence.slice(locus - halfBaseContextWindow, locus + halfBaseContextWindow + 1)
    val baseCounts = contigWindow.groupBy(identity).map(kv => (kv._1, kv._2.size))

    ReferenceContext(
      Bases.basesToString(contigSequence.slice(math.max(0, locus - flankingContextLength), locus)),
      Bases.basesToString(contigSequence.slice(locus + 1, math.min(contigSequence.length - 1, locus + flankingContextLength + 1))),
      baseCounts
    )
  }
}

case class VariantEvidence(contig: ContigName,
                           locus: Locus,
                           refBases: String,
                           altBases: String,
                           tumorReferenceEvidence: AlleleEvidence,
                           tumorVariantEvidence: AlleleEvidence,
                           normalReferenceEvidence: AlleleEvidence,
                           normalVariantEvidence: AlleleEvidence,
                           referenceContext: ReferenceContext) {

  def toVector = {
    DenseVector.vertcat(
      tumorReferenceEvidence.toDenseVector,
      tumorVariantEvidence.toDenseVector,
      normalReferenceEvidence.toDenseVector,
      normalVariantEvidence.toDenseVector
    )
  }
}

object VariantEvidence {

  def apply(tumorPileup: Pileup, normalPileup: Pileup): VariantEvidence = {

    val referenceAllele = Allele(tumorPileup.referenceBase, tumorPileup.referenceBase)

    val contigName = tumorPileup.contigName
    val locus = tumorPileup.locus

    val tumorDepth = tumorPileup.depth
    val variantAlleleFractions: Map[Allele, Double] =
      tumorPileup
        .elements
        .withFilter(_.allele.isSnv)
        .map(_.allele)
        .groupBy(identity)
        .map(kv => kv._1 -> kv._2.size / tumorDepth.toDouble )

    val referenceGenotype = Genotype(Map(referenceAllele -> 1.0))

    val mostFrequentVariantAllele = variantAlleleFractions.maxBy(_._2)
    val variantAllele =  mostFrequentVariantAllele._1
    val empiricalVariantAlleleFrequency =  math.min(0.01, mostFrequentVariantAllele._2)

    val germlineVariantGenotype =
      Genotype(
        Map(
          referenceAllele -> 0.5,
          mostFrequentVariantAllele._1 -> 0.5
        )
      )

    val somaticVariantGenotype =
      Genotype(
        Map(
          referenceAllele -> (1.0 - empiricalVariantAlleleFrequency),
          mostFrequentVariantAllele._1 -> empiricalVariantAlleleFrequency
        )
      )

    val likelihoods =
      (for {
        pileupAndVariantGenotype <- Vector(
          (normalPileup, germlineVariantGenotype),
          (tumorPileup, somaticVariantGenotype)
        )
        pileup = pileupAndVariantGenotype._1
        variantGenotype = pileupAndVariantGenotype._2
        likelihoods = Likelihood.likelihoodsOfGenotypes(
          pileup.elements,
          Array(referenceGenotype, variantGenotype),
          prior = Likelihood.uniformPrior,
          includeAlignment = false,
          logSpace = true,
          normalize = true
        )
      }
        yield pileup.sampleName -> likelihoods
        ).toMap

    val tumorSampleName = tumorPileup.sampleName
    val normalSampleName = normalPileup.sampleName

    VariantEvidence(
      contigName,
      locus,
      Bases.basesToString(variantAllele.refBases),
      Bases.basesToString(variantAllele.altBases),
      tumorReferenceEvidence = AlleleEvidence(likelihoods(tumorSampleName)(0), referenceAllele, tumorPileup),
      tumorVariantEvidence = AlleleEvidence(likelihoods(tumorSampleName)(1), variantAllele, tumorPileup),
      normalReferenceEvidence = AlleleEvidence(likelihoods(normalSampleName)(0), referenceAllele, normalPileup),
      normalVariantEvidence = AlleleEvidence(likelihoods(normalSampleName)(1), variantAllele, normalPileup),
      referenceContext = ReferenceContext(tumorPileup.contigSequence, locus, 3, 25)
    )
  }
}

  class ComputeSomaticEvidenceArgs
    extends Args
      with TumorNormalReadsArgs
      with PartitionedRegionsArgs
      with GenotypeFilterArguments
      with ReferenceArgs {

    @Args4jOption(name = "--dbsnp-vcf", required = false, usage = "VCF file to identify DBSNP variants")
    var dbSnpVcf: String = ""

    @Args4jOption(name = "--output", required = true)
    var output: String = ""

  }

  object ComputeSomaticEvidence extends SparkCommand[ComputeSomaticEvidenceArgs] {
    override val name = "compute-evidence"
    override val description = ""

    override def run(args: ComputeSomaticEvidenceArgs, sc: SparkContext) = {

      val sqlContext = new org.apache.spark.sql.SQLContext(sc)

      // this is used to implicitly convert an RDD to a DataFrame.
      import sqlContext.implicits._

      val reference = args.reference(sc)

      val (readsets, loci) = ReadSets(sc, args)
      val lociEvidence =
        computeLociEvidence(sc, args, reference, readsets, loci, args.minReadDepth, args.maxReadDepth)
        .toDF
      println(s"Found alternate bases at ${lociEvidence.count} / ${loci.count} negative training points")

      lociEvidence.printSchema()

      val flat =
        lociEvidence.select(
          $"contig",
          $"locus",
          $"refBases",
          $"altBases",
          $"tumorReferenceEvidence.*",
          $"tumorVariantEvidence.*",
          $"normalReferenceEvidence.*",
          $"normalVariantEvidence.*",
          $"referenceContext.leftContext",
          $"referenceContext.rightContext"
        )

      flat.printSchema()
      flat.write.format("com.databricks.spark.csv").mode(SaveMode.Overwrite).save(args.output)



    }

    def computeLociEvidence(sc: SparkContext,
                            args: ComputeSomaticEvidenceArgs,
                            reference: ReferenceBroadcast,
                            readsets: ReadSets,
                            lociSet: LociSet,
                            minReadDepth: Int,
                            maxReadDepth: Int) = {

      val partitionedReads =
        PartitionedRegions(
          readsets.allMappedReads,
          lociSet,
          args
        )

      val normalSampleName = args.normalSampleName
      val tumorSampleName = args.tumorSampleName

      pileupFlatMapTwoSamples(
        partitionedReads,
        sample1Name = normalSampleName,
        sample2Name = tumorSampleName,
        skipEmpty = true, // skip empty pileups
        function = (pileupNormal, pileupTumor) => {
          if (pileupNormal.depth > minReadDepth &&
            pileupNormal.depth < maxReadDepth &&
            pileupTumor.depth > minReadDepth &&
            pileupTumor.depth < maxReadDepth &&
            pileupTumor.referenceDepth != pileupTumor.depth) {
            val stats = computePileupStats(
              pileupTumor,
              pileupNormal
            )
            stats.iterator
          }
          else
            Iterator.empty
        },
        reference = reference
      )
    }

    def computePileupStats(tumorPileup: Pileup,
                           normalPileup: Pileup): Option[VariantEvidence] = {

      // For now, we skip loci that have no reads mapped. We may instead want to emit NoCall in this case.
      if (tumorPileup.elements.isEmpty
        || normalPileup.elements.isEmpty
        || tumorPileup.referenceDepth == tumorPileup.depth // skip computation if no alternate reads
        // Skip if the pileup contains variants that are not SNV
        || tumorPileup.elements.count(_.allele.isVariant) != tumorPileup.elements.count(_.allele.isSnv)
      )
        return None

      Some(
        VariantEvidence(
          tumorPileup = tumorPileup,
          normalPileup = normalPileup
        )
      )
  }
}
