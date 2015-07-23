/**
* Hub Miner: a hubness-aware machine learning experimentation library.
* Copyright (C) 2014  Nenad Tomasev. Email: nenad.tomasev at gmail.com
* 
* This program is free software: you can redistribute it and/or modify it under
* the terms of the GNU General Public License as published by the Free Software
* Foundation, either version 3 of the License, or (at your option) any later
* version.
* 
* This program is distributed in the hope that it will be useful, but WITHOUT
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
* FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with
* this program. If not, see <http://www.gnu.org/licenses/>.
*/
package distances.secondary.snd;

import data.neighbors.NeighborSetFinder;
import data.neighbors.SharedNeighborFinder;
import data.neighbors.hubness.HubnessExtremesGrabber;
import data.neighbors.hubness.HubnessSkewAndKurtosisExplorer;
import data.neighbors.hubness.HubnessAboveThresholdExplorer;
import data.neighbors.hubness.HubnessVarianceExplorer;
import data.neighbors.hubness.KNeighborEntropyExplorer;
import data.neighbors.hubness.TopHubsClusterUtil;
import data.representation.DataSet;
import data.representation.discrete.DiscretizedDataSet;
import distances.primary.CombinedMetric;
import distances.primary.DistanceMeasure;
import distances.sparse.SparseCombinedMetric;
import distances.sparse.SparseMetric;
import ioformat.FileUtil;
import ioformat.IOARFF;
import ioformat.IOCSV;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import learning.unsupervised.evaluation.quality.QIndexSilhouette;
import util.BasicMathUtil;

/**
 * A utility batch analyzer for shared-neighbor distance effectiveness on a
 * specified list of datasets with the specified primary metrics.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SharedNeighborDSAnalyzer {

    int kMax;
    // Noise and mislabeling levels to vary.
    float noiseMin, noiseMax, noiseStep, mlMin, mlMax, mlStep;
    // Directory structure for input and output.
    File inConfigFile, inDir, outDir, currOutDSDir;
    // Paths and metric objects.
    ArrayList<String> dsPaths = new ArrayList<>(100);
    ArrayList<CombinedMetric> dsMetric = new ArrayList<>(100);
    // Original dataset and the current one (after applying some modifications).
    DataSet originalDSet, currDSet;
    DiscretizedDataSet currDiscDset;
    // The current metric object.
    CombinedMetric currCmet;
    // Original label array.
    int[] originalLabels;
    // Number of categories in the data.
    int numCategories;
    // Shared-neighbor metric parameters.
    boolean hubnessWeightedSND = true;
    float thetaSimhub = 0;
    int kSND = 50;

    /**
     *
     * @param inConfigFile File that contains the experiment configuration.
     */
    public SharedNeighborDSAnalyzer(File inConfigFile) {
        this.inConfigFile = inConfigFile;
    }

    /**
     * This method runs all the experiments.
     *
     * @throws Exception
     */
    public void runAllTests() throws Exception {
        int counter = 0;
        // For each dataset.
        for (String dsPath : dsPaths) {
            // Load the data.
            File dsFile = new File(dsPath);
            // Currently it has to be specified whether the data is in sparse
            // format or not. If it is, a prefix of "sparse:" is prepended to
            // the specified path.
            if (dsPath.startsWith("sparse:")) {
                String trueDSPath = dsPath.substring(dsPath.indexOf(':') + 1,
                        dsPath.length());
                IOARFF pers = new IOARFF();
                originalDSet = pers.loadSparse(trueDSPath);
            } else {
                if (dsPath.endsWith(".csv")) {
                    IOCSV reader = new IOCSV(true, ",");
                    originalDSet = reader.readData(dsFile);
                } else if (dsPath.endsWith(".arff")) {
                    IOARFF persister = new IOARFF();
                    originalDSet = persister.load(dsPath);
                } else {
                    System.out.println("error, could not read: " + dsPath);
                    continue;
                }
            }
            // Inform the user of the dataset the current tests are running on.
            System.out.println("testing on: " + dsPath);
            originalDSet.standardizeCategories();
            originalLabels = originalDSet.obtainLabelArray();
            numCategories = originalDSet.countCategories();
            int memCleanCount = 0;
            // Go through all the noise and mislabeling levels that were
            // specified in the configuration file. No noise and no mislabeling
            // is also an option, a default one at that.
            for (float noise = noiseMin; noise <= noiseMax; noise +=
                    noiseStep) {
                for (float ml = mlMin; ml <= mlMax; ml += mlStep) {
                    if (++memCleanCount % 5 == 0) {
                        System.gc();
                    }
                    currDSet = originalDSet.copy();
                    if (ml > 0) {
                        currDSet.induceMislabeling(ml, numCategories);
                    }
                    if (noise > 0) {
                        currDSet.addGaussianNoiseToNormalizedCollection(
                                noise, 0.1f);
                    }
                    if (hubnessWeightedSND) {
                        currOutDSDir = new File(outDir,
                                dsFile.getName().substring(
                                0, dsFile.getName().lastIndexOf("."))
                                + "SNH" + this.thetaSimhub + File.separator
                                + "k" + kMax + File.separator + "ml" + ml
                                + File.separator + "noise" + noise);
                    } else {
                        currOutDSDir = new File(outDir,
                                dsFile.getName().substring(0,
                                dsFile.getName().lastIndexOf("."))
                                + "SN" + File.separator + "k" + kMax
                                + File.separator + "ml" + ml + File.separator
                                + "noise" + noise);
                    }
                    FileUtil.createDirectory(currOutDSDir);
                    currCmet = dsMetric.get(counter);
                    float[] cP = currDSet.getClassPriors();

                    // Perform initial calculations.

                    NeighborSetFinder nsfTemp = new NeighborSetFinder(
                            currDSet, currCmet);
                    nsfTemp.calculateDistances();
                    nsfTemp.calculateNeighborSetsMultiThr(kSND, 6);

                    float[][] dMatPrimary = nsfTemp.getDistances();
                    // Primary distance analysis.
                    float maxPrimary = 0;
                    float minPrimary = Float.MAX_VALUE;
                    for (int i = 0; i < dMatPrimary.length; i++) {
                        for (int j = 0; j < dMatPrimary[i].length; j++) {
                            maxPrimary = Math.max(maxPrimary,
                                    dMatPrimary[i][j]);
                            minPrimary = Math.min(minPrimary,
                                    dMatPrimary[i][j]);
                        }
                    }
                    double intraSumPrimary = 0;
                    double interSumPrimary = 0;
                    double intraNumPrimary = 0;
                    double interNumPrimary = 0;
                    // The distribution of intra- and inter-class primary
                    // distances, with 50 bins.
                    double[] intraDistrPrimary = new double[50];
                    double[] interDistrPrimary = new double[50];
                    for (int i = 0; i < dMatPrimary.length; i++) {
                        for (int j = 0; j < dMatPrimary[i].length; j++) {
                            // Re-scale the primary distances.
                            dMatPrimary[i][j] = (dMatPrimary[i][j]
                                    - minPrimary) / (maxPrimary - minPrimary);
                            if (currDSet.getLabelOf(i) == currDSet.
                                    getLabelOf(i + j + 1)) {
                                intraSumPrimary += dMatPrimary[i][j];
                                intraNumPrimary++;
                                if (dMatPrimary[i][j] < 1) {
                                    intraDistrPrimary[(int)
                                            (dMatPrimary[i][j] * 50)]++;
                                } else {
                                    intraDistrPrimary[49]++;
                                }
                            } else {
                                interSumPrimary += dMatPrimary[i][j];
                                interNumPrimary++;
                                if (dMatPrimary[i][j] < 1) {
                                    interDistrPrimary[(int)
                                            (dMatPrimary[i][j] * 50)]++;
                                } else {
                                    interDistrPrimary[49]++;
                                }
                            }
                        }
                    }
                    // Average inter- and intra- class primary distances.
                    double interAvgPrimary =
                            interSumPrimary / interNumPrimary;
                    double intraAvgPrimary =
                            intraSumPrimary / intraNumPrimary;
                    double avgDPrimary = interSumPrimary + intraSumPrimary;
                    avgDPrimary /= (interNumPrimary + intraNumPrimary);
                    // Scale by the overall average distance.
                    double interAvgRatioPrimary =
                            interAvgPrimary / avgDPrimary;
                    double intraAvgRatioPrimary =
                            intraAvgPrimary / avgDPrimary;
                    for (int i = 0; i < 50; i++) {
                        interDistrPrimary[i] /= interNumPrimary;
                        intraDistrPrimary[i] /= intraNumPrimary;
                    }
                    // Initialize the shared-neighbor finder.
                    SharedNeighborFinder snf =
                            new SharedNeighborFinder(nsfTemp);
                    if (hubnessWeightedSND) {
                        snf.obtainWeightsFromHubnessInformation(thetaSimhub);
                    }
                    // Count the kNN set intersections.
                    snf.countSharedNeighbors();
                    // First get the similarity matrix from the SNN-s.
                    float[][] simMatSNN = snf.getSharedNeighborCounts();
                    // Now transform it to a distance matrix.
                    float[][] dMatSecondary = new float[simMatSNN.length][];
                    for (int i = 0; i < dMatSecondary.length; i++) {
                        dMatSecondary[i] = new float[simMatSNN[i].length];
                        for (int j = 0; j < dMatSecondary[i].length; j++) {
                            dMatSecondary[i][j] = kSND - simMatSNN[i][j];
                        }
                    }
                    // Normalize the scores and get the intra- and
                    // inter-class distributions and averages.
                    float max = 0;
                    float min = Float.MAX_VALUE;
                    for (int i = 0; i < dMatSecondary.length; i++) {
                        for (int j = 0; j < dMatSecondary[i].length; j++) {
                            max = Math.max(max, dMatSecondary[i][j]);
                            min = Math.min(min, dMatSecondary[i][j]);
                        }
                    }
                    double intraSum = 0;
                    double interSum = 0;
                    double intraNum = 0;
                    double interNum = 0;
                    double[] intraDistr = new double[50];
                    double[] interDistr = new double[50];
                    // The same analysis as in the primary case above.
                    for (int i = 0; i < dMatSecondary.length; i++) {
                        for (int j = 0; j < dMatSecondary[i].length; j++) {
                            dMatSecondary[i][j] = (dMatSecondary[i][j]
                                    - min) / (max - min);
                            if (currDSet.getLabelOf(i)
                                    == currDSet.getLabelOf(i + j + 1)) {
                                intraSum += dMatSecondary[i][j];
                                intraNum++;
                                if (dMatSecondary[i][j] < 1) {
                                    intraDistr[(int)
                                            (dMatSecondary[i][j] * 50)]++;
                                } else {
                                    intraDistr[49]++;
                                }
                            } else {
                                interSum += dMatSecondary[i][j];
                                interNum++;
                                if (dMatSecondary[i][j] < 1) {
                                    interDistr[(int) (dMatSecondary[i][j]
                                            * 50)]++;
                                } else {
                                    interDistr[49]++;
                                }
                            }
                        }
                    }
                    double interAvg = interSum / interNum;
                    double intraAvg = intraSum / intraNum;
                    double avgD = interSum + intraSum;
                    avgD /= (interNum + intraNum);
                    // Normalize by the overall average distance.
                    double interAvgRatio = interAvg / avgD;
                    double intraAvgRatio = intraAvg / avgD;
                    for (int i = 0; i < 50; i++) {
                        interDistr[i] /= interNum;
                        intraDistr[i] /= intraNum;
                    }
                    // Make a distance calculator out of snf.
                    SharedNeighborCalculator snc;
                    if (hubnessWeightedSND) {
                        snc = new SharedNeighborCalculator(snf,
                                SharedNeighborCalculator.
                                WeightingType.HUBNESS_INFORMATION);
                    } else {
                        snc = new SharedNeighborCalculator(snf,
                                SharedNeighborCalculator.WeightingType.NONE);
                    }

                    NeighborSetFinder nsf = new NeighborSetFinder(
                            currDSet, dMatSecondary, snc);
                    nsf.calculateNeighborSetsMultiThr(kMax, 6);

                    // Silhouette index is often used in cluster analysis
                    // and here we use it to compare how well the data is
                    // separated into clusters in the primary and the
                    // secondary metric space.
                    // First we look at the SNN metric space.
                    QIndexSilhouette silIndex = new QIndexSilhouette(
                            currDSet.countCategories(),
                            currDSet.obtainLabelArray(), currDSet);
                    silIndex.setDistanceMatrix(dMatSecondary);
                    silIndex.hubnessArray = nsf.getNeighborFrequencies();
                    float silData = silIndex.validity();
                    // Here we look at the primary metric space.
                    silIndex = new QIndexSilhouette(
                            currDSet.countCategories(),
                            currDSet.obtainLabelArray(), currDSet);
                    silIndex.setDistanceMatrix(dMatPrimary);
                    silIndex.hubnessArray = nsfTemp.getNeighborFrequencies();
                    float silDataPrimary = silIndex.validity();

                    // Now we analyze the hubness-related properties of the
                    // data.
                    HubnessAboveThresholdExplorer hte =
                            new HubnessAboveThresholdExplorer(1, true, nsf);
                    HubnessSkewAndKurtosisExplorer hske =
                            new HubnessSkewAndKurtosisExplorer(nsf);
                    HubnessExtremesGrabber heg =
                            new HubnessExtremesGrabber(true, nsf);
                    HubnessVarianceExplorer hve =
                            new HubnessVarianceExplorer(nsf);
                    TopHubsClusterUtil thcu =
                            new TopHubsClusterUtil(nsf);
                    KNeighborEntropyExplorer knee =
                            new KNeighborEntropyExplorer(nsf,
                            numCategories);

                    // Without going into much detail here, the semantics of
                    // the data collected here will be clear from the
                    // following print-outs.
                    float[] aboveZeroArray =
                            hte.getThresholdPercentageArray();
                    hske.calcSkewAndKurtosisArrays();
                    float[] skewArray = hske.getOccFreqsSkewnessArray();
                    float[] kurtosisArray = hske.getOccFreqsKurtosisArray();
                    float[][] highestHubnesses =
                            heg.getHubnessExtremesForKValues(15);
                    float[] stDevArray = hve.getStDevForKRange();
                    thcu.calcTopHubnessDiamAndAvgDist(10);
                    float[] topHubClustDiamsArr =
                            thcu.getTopHubClusterDiameters();
                    float[] topHubClustAvgDistArr =
                            thcu.getTopHubClusterAvgDists();
                    knee.calculateAllKNNEntropyStats();
                    float[] kEntropiesMeans =
                            knee.getDirectEntropyMeans();
                    float[] kHubnessEntropiesMeans =
                            knee.getReverseEntropyMeans();
                    float[] kEntropiesStDevs =
                            knee.getDirectEntropyStDevs();
                    float[] kHubnessEntropiesStDevs =
                            knee.getReverseEntropyStDevs();
                    float[] kEntropiesSkews =
                            knee.getDirectEntropySkews();
                    float[] kHubnessEntropiesSkews =
                            knee.getReverseEntropySkews();
                    float[] kEntropiesKurtosis =
                            knee.getDirectEntropyKurtosisVals();
                    float[] kHubnessEntropiesKurtosis =
                            knee.getReverseEntropyKurtosisVals();
                    float[] entDiffs =
                            knee.getAverageDirectAndReverseEntropyDifs();
                    float[] bhArray = nsf.getLabelMismatchPercsAllK();
                    float[][][] gCtoChubness = new float[kMax][][];
                    for (int k = 1; k <= kMax; k++) {
                        gCtoChubness[k - 1] =
                                nsf.getGlobalClassToClassForKforFuzzy(
                                k, numCategories, 0.01f, true);
                    }

                    File currOutFile = new File(currOutDSDir,
                            "hubnessOverview.txt");
                    PrintWriter pw = new PrintWriter(
                            new FileWriter(currOutFile));
                    try {
                        pw.println("dataset: " + dsFile.getName());
                        pw.println("k_max: " + kMax);
                        pw.println("shared neighbor k: " + kSND);
                        pw.println("noise: " + noise);
                        pw.println("ml: " + ml);
                        pw.println("instances: " + originalDSet.size());
                        pw.println("numCat: " + numCategories);
                        pw.print("class priors: ");
                        for (int i = 0; i < numCategories; i++) {
                            pw.print(BasicMathUtil.makeADecimalCutOff(cP[i],
                                    3));
                            pw.print(" ");
                        }
                        pw.println();
                        try {
                            pw.println("dim: "
                                    + originalDSet.fAttrNames.length);
                        } catch (Exception e) {
                            System.err.println(e.getMessage());
                        }
                        pw.println("intra-class avg normalized dist: "
                                + intraAvgPrimary);
                        pw.println("inter-class avg normalized dist: "
                                + interAvgPrimary);
                        pw.println("intra-class avg normalized SN dist: "
                                + intraAvg);
                        pw.println("inter-class avg normalized SN dist: "
                                + interAvg);
                        pw.println("intra-class avg normalized by "
                                + "AVG dist: " + intraAvgRatioPrimary);
                        pw.println("inter-class avg normalized by "
                                + "AVG dist: " + interAvgRatioPrimary);
                        pw.println("intra-class avg normalized by "
                                + "AVG SN dist: " + intraAvgRatio);
                        pw.println("inter-class avg normalized by "
                                + "AVG SN dist: " + interAvgRatio);
                        pw.println("original silIndex: " + silDataPrimary);
                        pw.println("SN silIndex: " + silData);
                        pw.println("-------------------------------------");
                        pw.println("distance distributions: ");
                        pw.print("intra-class ");
                        for (int i = 0; i < 50; i++) {
                            pw.print(intraDistrPrimary[i] + " ");
                        }
                        pw.println();
                        pw.print("inter-class ");
                        for (int i = 0; i < 50; i++) {
                            pw.print(interDistrPrimary[i] + " ");
                        }
                        pw.println();
                        pw.print("intra-class SND ");
                        for (int i = 0; i < 50; i++) {
                            pw.print(intraDistr[i] + " ");
                        }
                        pw.println();
                        pw.print("inter-class SND ");
                        for (int i = 0; i < 50; i++) {
                            pw.print(interDistr[i] + " ");
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("stDevArray: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                stDevArray[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    stDevArray[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("skewArray: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                skewArray[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    skewArray[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kurtosisArray: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kurtosisArray[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kurtosisArray[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("bad hubness: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                bhArray[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    bhArray[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEntropyMeans: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kEntropiesMeans[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesMeans[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEntropyStDevs: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kEntropiesStDevs[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesStDevs[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEntropySkews: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kEntropiesSkews[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesSkews[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEntropyKurtosis: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kEntropiesKurtosis[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kEntropiesKurtosis[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kHubnessEntropyMeans: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kHubnessEntropiesMeans[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kHubnessEntropiesMeans[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kHubnessEntropyStDevs: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kHubnessEntropiesStDevs[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kHubnessEntropiesStDevs[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kHubnessEntropySkews: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kHubnessEntropiesSkews[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kHubnessEntropiesSkews[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kHubnessEntropyKurtosis: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                kHubnessEntropiesKurtosis[0], 3));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    kHubnessEntropiesKurtosis[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("kEnt - khEnt avgs: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                entDiffs[0], 3));
                        for (int i = 1; i < entDiffs.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    entDiffs[i], 3));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("hubness above zero percentage Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        hte = new HubnessAboveThresholdExplorer(2, true, nsf);
                        aboveZeroArray = hte.getThresholdPercentageArray();
                        pw.println("-------------------------------------");
                        pw.println("hubness above one percentage Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        hte = new HubnessAboveThresholdExplorer(3, true, nsf);
                        aboveZeroArray = hte.getThresholdPercentageArray();
                        pw.println("-------------------------------------");
                        pw.println("hubness above two percentage Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        hte = new HubnessAboveThresholdExplorer(4, true, nsf);
                        aboveZeroArray = hte.getThresholdPercentageArray();
                        pw.println("-------------------------------------");
                        pw.println("hubness above three percentage "
                                + "Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        hte = new HubnessAboveThresholdExplorer(5, true, nsf);
                        aboveZeroArray = hte.getThresholdPercentageArray();
                        pw.println("-------------------------------------");
                        pw.println("hubness above four percentage Array: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                aboveZeroArray[0], 2));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print(","
                                    + BasicMathUtil.makeADecimalCutOff(
                                    aboveZeroArray[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("Top ten hubs diam: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                topHubClustDiamsArr[0], 2));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    topHubClustDiamsArr[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("Top ten hubs avg within cluster dist: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                topHubClustAvgDistArr[0], 2));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    topHubClustAvgDistArr[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        thcu.calcTopHubnessDiamAndAvgDist(5);
                        topHubClustDiamsArr = thcu.getTopHubClusterDiameters();
                        topHubClustAvgDistArr = thcu.getTopHubClusterAvgDists();
                        pw.println("Top five hubs diam: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                topHubClustDiamsArr[0], 2));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    topHubClustDiamsArr[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("Top five hubs avg within cluster dist: ");
                        pw.print(BasicMathUtil.makeADecimalCutOff(
                                topHubClustAvgDistArr[0], 2));
                        for (int i = 1; i < stDevArray.length; i++) {
                            pw.print("," + BasicMathUtil.makeADecimalCutOff(
                                    topHubClustAvgDistArr[i], 2));
                        }
                        pw.println();
                        pw.println("-------------------------------------");
                        pw.println("highest hubnesses (each line is for "
                                + "one k value, lines go from zero to"
                                + " k_max): ");
                        for (int k = 0; k < kMax; k++) {
                            pw.print("k: " + (k + 1) + ":: ");
                            pw.print(BasicMathUtil.makeADecimalCutOff(
                                    highestHubnesses[k][0], 3));
                            for (int i = 1; i < 15; i++) {
                                pw.print(","
                                        + BasicMathUtil.makeADecimalCutOff(
                                        highestHubnesses[k][i], 3));
                            }
                            pw.println();
                        }
                        pw.println("-------------------------------------");
                        pw.println("global class to class hubness matrices"
                                + " for all K-s: ");
                        for (int k = 1; k <= kMax; k++) {
                            pw.println("k = " + k);
                            for (int c1 = 0; c1 < numCategories; c1++) {
                                for (int c2 = 0; c2 < numCategories; c2++) {
                                    pw.print(
                                            BasicMathUtil.
                                            makeADecimalCutOff(
                                            gCtoChubness[k - 1][c1][c2], 3));
                                    pw.print(" ");
                                }
                                pw.println();
                            }
                            pw.println();
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                        System.out.println(e.getMessage());
                    } finally {
                        pw.close();
                    }
                }
            }
            counter++;
        }
    }

    /**
     * Loads all the experiment parameters from a file that was specified in the
     * constructor.
     *
     * @throws Exception
     */
    public void loadParameters() throws Exception {
        BufferedReader br = new BufferedReader(
                new InputStreamReader(new FileInputStream(inConfigFile)));
        try {
            String s = br.readLine();
            String[] lineItems;
            // Primary integer and float metrics.
            Class currIntMet;
            Class currFloatMet;

            while (s != null) {
                s = s.trim();
                if (s.startsWith("@in_directory")) {
                    lineItems = s.split(" ");
                    inDir = new File(lineItems[1]);
                } else if (s.startsWith("@shared_neighbors_metric_override")) {
                    lineItems = s.split(" ");
                    kSND = Integer.parseInt(lineItems[1]);
                    hubnessWeightedSND = Boolean.parseBoolean(lineItems[2]);
                    if (hubnessWeightedSND) {
                        thetaSimhub = Float.parseFloat(lineItems[3]);
                    }
                } else if (s.startsWith("@out_directory")) {
                    lineItems = s.split(" ");
                    outDir = new File(lineItems[1]);
                } else if (s.startsWith("@k_max")) {
                    lineItems = s.split(" ");
                    kMax = Integer.parseInt(lineItems[1]);
                } else if (s.startsWith("@noise_range")) {
                    lineItems = s.split(" ");
                    noiseMin = Float.parseFloat(lineItems[1]);
                    noiseMax = Float.parseFloat(lineItems[2]);
                    noiseStep = Float.parseFloat(lineItems[3]);
                } else if (s.startsWith("@mislabeled_range")) {
                    lineItems = s.split(" ");
                    mlMin = Float.parseFloat(lineItems[1]);
                    mlMax = Float.parseFloat(lineItems[2]);
                    mlStep = Float.parseFloat(lineItems[3]);
                } else if (s.startsWith("@dataset")) {
                    lineItems = s.split(" ");
                    dsPaths.add(lineItems[1]);
                    if (lineItems[1].startsWith("sparse:")) {
                        // We have to indicate that the data is in the sparse
                        // format and that we consequently need a sparse metric
                        // object to handle it.
                        SparseCombinedMetric smc =
                                new SparseCombinedMetric(
                                null, null, (SparseMetric) (Class.forName(
                                lineItems[2]).newInstance()),
                                CombinedMetric.DEFAULT);
                        dsMetric.add(smc);
                    } else {
                        CombinedMetric cmet = new CombinedMetric();
                        if (!lineItems[2].equals("null")) {
                            currIntMet = Class.forName(lineItems[2]);
                            cmet.setIntegerMetric(
                                    (DistanceMeasure)
                                    (currIntMet.newInstance()));
                        }
                        if (!lineItems[3].equals("null")) {
                            currFloatMet = Class.forName(lineItems[3]);
                            cmet.setFloatMetric(
                                    (DistanceMeasure)
                                    (currFloatMet.newInstance()));
                        }
                        cmet.setCombinationMethod(CombinedMetric.DEFAULT);
                        dsMetric.add(cmet);
                    }
                }
                s = br.readLine();
            }
            // Now complete/correct the file paths by prepending the directory.
            for (int i = 0; i < dsPaths.size(); i++) {
                if (!dsPaths.get(i).startsWith("sparse:")) {
                    dsPaths.set(i, (new File(inDir, dsPaths.get(i))).getPath());
                } else {
                    dsPaths.set(i, "sparse:" + (new File(
                            inDir, dsPaths.get(i).substring(
                            dsPaths.get(i).indexOf(":") + 1,
                            dsPaths.get(i).length()))).getPath());
                }
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            throw e;
        } finally {
            br.close();
        }
    }

    /**
     * Runs all the experiments specified in the configuration file for
     * comparing the properties of primary distances and secondary SNN distances
     * on a series of datasets under a series of conditions.
     *
     * @param args One command line argument - the path to the configuration
     * file.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("Runs all the experiments specified in the "
                    + "configuration file for comparing the properties of "
                    + "primary distances and secondary SNN distances on a "
                    + "series of datasets under a series of conditions.");
            System.out.println("---------------------------------------------");
            System.out.println("1 parameter - file with test configuration");
            return;
        }
        File inConfigFile = new File(args[0]);
        SharedNeighborDSAnalyzer tester =
                new SharedNeighborDSAnalyzer(inConfigFile);
        tester.loadParameters();
        tester.runAllTests();
    }
}
