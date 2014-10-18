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
package data.neighbors.hubness.experimental;

import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import feature.correlation.PearsonCorrelation;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import util.CommandLineParser;

/**
 * This utility experimental script generates Gaussian mixture datasets of the
 * specified dimensionality and calculates the correlations between norm and
 * hubness/gb-hubness and density, where gb-hubness denotes the good-bad hubness
 * ratio.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MultiGaussianLocalityExplorer {

    private int numDatatets;
    private int dim;
    private int numClasses;
    private int kMin = 1;
    private int kMax = 20;
    private float[][] featureStDevs;
    private float[][] featureMeans;
    private DataSet dataset;
    private File outDir;
    private File[] dataDirs;
    private int catMinSize = 50;
    private int catMaxSize = 400;
    private float[][] distances;
    private float[] pointNorms;

    /**
     * This method runs all the experiments, generates the synthetic datasets,
     * calculates the norm, the hubness, the density - and correlates them.
     *
     * @throws Exception
     */
    public void runExperiments() throws Exception {
        Random randa = new Random();
        // Initialize output data directories.
        dataDirs = new File[numDatatets];
        for (int dsetIndex = 0; dsetIndex < numDatatets; dsetIndex++) {
            dataDirs[dsetIndex] = new File(outDir, "ds" + dsetIndex);
        }
        CombinedMetric cmet = CombinedMetric.FLOAT_EUCLIDEAN;
        // Generate all the synthetic distributions.
        // These parameters are used for generating the overlapping Gaussian
        // distributions in the high-dimensional space.
        float alpha = 0.75f;
        float beta = 1.5f;
        float choice;
        float compensator = 0.75f;
        for (int dsetIndex = 0; dsetIndex < numDatatets; dsetIndex++) {
            // Generate the feature value distributions.
            featureStDevs = new float[numClasses][dim];
            featureMeans = new float[numClasses][dim];
            // Start with one class.
            for (int d = 0; d < dim; d++) {
                featureMeans[0][d] = 0;
                featureStDevs[0][d] =
                        (float) Math.max(0.1, 4 + randa.nextGaussian());
            }
            // Iterate over the remaining classes.
            for (int catIndex = 1; catIndex < numClasses; catIndex++) {
                // Choose an existing class distribution to overlap with.
                int brother = choosePairedDistribution(catIndex, randa);
                for (int d = 0; d < dim; d++) {
                    choice = randa.nextFloat();
                    // The sign is randomized.
                    if (choice < 0.5) {
                        featureMeans[catIndex][d] = featureMeans[brother][d]
                                + compensator * featureStDevs[brother][d];
                    } else {
                        featureMeans[catIndex][d] = featureMeans[brother][d]
                                - compensator * featureStDevs[brother][d];
                    }
                    featureStDevs[catIndex][d] =
                            alpha * featureStDevs[brother][d] + (beta - alpha)
                            * randa.nextFloat() * featureStDevs[brother][d];
                }
            }
            // Persist the generative model.
            File outDistrFile = new File(dataDirs[dsetIndex],
                    "generatingModel.txt");
            FileUtil.createFile(outDistrFile);
            PrintWriter pw = new PrintWriter(new FileWriter(outDistrFile));
            try {
                for (int c = 0; c < numClasses; c++) {
                    pw.println("category: " + c);
                    pw.print("means: " + featureMeans[c][0]);
                    for (int j = 1; j < dim; j++) {
                        pw.print("," + featureMeans[c][j]);
                    }
                    pw.println();
                    pw.print("stDevs: " + featureStDevs[c][0]);
                    for (int j = 1; j < dim; j++) {
                        pw.print("," + featureStDevs[c][j]);
                    }
                    pw.println();
                }
            } catch (Exception e) {
                throw e;
            } finally {
                pw.close();
            }
            // Generate the synthetic dataset.
            int size;
            DataInstance instance;
            dataset = new DataSet();
            // Initialize the feature names.
            dataset.fAttrNames = new String[dim];
            for (int d = 0; d < dim; d++) {
                dataset.fAttrNames[d] = "att" + d;
            }
            dataset.data = new ArrayList<>(numClasses * catMaxSize);
            // Iterate over the categories.
            for (int catIndex = 0; catIndex < numClasses; catIndex++) {
                // Determine the class size.
                size = catMinSize + (int) ((catMaxSize - catMinSize)
                        * randa.nextFloat());
                System.out.println("cSize " + catIndex + ":" + size);
                // Generate the synthetic data instances.
                for (int i = 0; i < size; i++) {
                    instance = new DataInstance(dataset);
                    dataset.addDataInstance(instance);
                    instance.embedInDataset(dataset);
                    instance.setCategory(catIndex);
                    for (int d = 0; d < dim; d++) {
                        instance.fAttr[d] = featureMeans[catIndex][d]
                                + featureStDevs[catIndex][d]
                                * (float) randa.nextGaussian();
                    }
                }
            }
            System.out.println("Data size: " + dataset.size());
            // Persist the synthetic dataset.
            IOARFF persister = new IOARFF();
            persister.saveLabeledWithIdentifiers(dataset,
                    (new File(dataDirs[dsetIndex], "dataset"
                    + dsetIndex + ".arff")).getPath(), null);
            // Calculate the distance matrix.
            distances = dataset.calculateDistMatrixMultThr(cmet, 4);
            float maxDist = 0;
            for (int i = 0; i < distances.length; i++) {
                for (int j = 0; j < distances[i].length; j++) {
                    if (distances[i][j] > maxDist) {
                        maxDist = distances[i][j];
                    }
                }
            }
            // Calculate the k-nearest neighbor sets.
            NeighborSetFinder nsf = new NeighborSetFinder(
                    dataset, distances, cmet);
            nsf.calculateNeighborSetsMultiThr(kMax, 4);
            // Correlate norm with hubness, gb-hubness and density.
            float[] correlationBetweenNormAndHubness =
                    new float[kMax - kMin + 1];
            float[] correlationBetweenNormAndGBHMeasure =
                    new float[kMax - kMin + 1];
            float[] correlationBetweenNormAndDensity =
                    new float[kMax - kMin + 1];
            // Calculate point centrality for all points.
            MinkowskiMetric cmetEuc = new MinkowskiMetric();
            pointNorms = new float[dataset.size()];
            for (int index = 0; index < dataset.size(); index++) {
                pointNorms[index] = Float.MAX_VALUE;
                for (int catIndex = 0; catIndex < numClasses; catIndex++) {
                    pointNorms[index] = Math.min(cmetEuc.dist(
                            featureMeans[catIndex],
                            dataset.data.get(index).fAttr) / maxDist,
                            pointNorms[index]);
                }
            }
            // Calculate the correlations over the entire neighborhood size
            // range.
            for (int k = kMin; k <= kMax; k++) {
                nsf.recalculateStatsForSmallerK(k);
                float[] neighbOccFreqs = nsf.getFloatOccFreqs();
                double[] density = nsf.getDataDensitiesByNormalizedRadius();
                correlationBetweenNormAndHubness[k - kMin] =
                        PearsonCorrelation.correlation(neighbOccFreqs,
                        pointNorms);
                int[] ghArray = nsf.getGoodFrequencies();
                int[] bhArray = nsf.getBadFrequencies();
                float[] gbhWeightedArray = new float[ghArray.length];
                float[] weights =
                        nsf.getSimhubAlternateWeightsGoodnessProportional();
                for (int i = 0; i < gbhWeightedArray.length; i++) {
                    gbhWeightedArray[i] = (ghArray[i] + bhArray[i])
                            * weights[i];
                }
                correlationBetweenNormAndGBHMeasure[k - kMin] =
                        PearsonCorrelation.correlation(gbhWeightedArray,
                        pointNorms);
                correlationBetweenNormAndDensity[k - kMin] =
                        PearsonCorrelation.correlation(pointNorms, density);
            }
            File correlationFile = new File(dataDirs[dsetIndex],
                    "correlations" + dsetIndex + ".csv");
            FileUtil.createFile(correlationFile);
            // Print out the correlations to a file.
            pw = new PrintWriter(new FileWriter(correlationFile));
            try {
                pw.print("corr(norm, -) for k:");
                for (int k = kMin; k <= kMax; k++) {
                    pw.print("," + k);
                }
                pw.println();
                pw.print("hubness");
                pw.print("," + correlationBetweenNormAndHubness[0]);
                for (int k = kMin + 1; k <= kMax; k++) {
                    pw.print("," + correlationBetweenNormAndHubness[k - kMin]);
                }
                pw.println();
                pw.print("gbh measure");
                pw.print("," + correlationBetweenNormAndGBHMeasure[0]);
                for (int k = kMin + 1; k <= kMax; k++) {
                    pw.print(","
                            + correlationBetweenNormAndGBHMeasure[k - kMin]);
                }
                pw.println();
                pw.print("density");
                pw.print("," + correlationBetweenNormAndDensity[0]);
                for (int k = kMin + 1; k <= kMax; k++) {
                    pw.print("," + correlationBetweenNormAndDensity[k - kMin]);
                }
                pw.println();
            } catch (Exception e) {
                throw e;
            } finally {
                pw.close();
            }

        }
    }

    /**
     * This method chooses an index of the paired distribution to match the
     * current one with.
     *
     * @param currIndex Integer that is the index of the current distribution.
     * @param randa Random number generator.
     * @return Integer that is the index of the distribution to pair the
     * original one with.
     */
    private static int choosePairedDistribution(int currIndex, Random randa) {
        if (currIndex == 1) {
            return 0;
        } else {
            return randa.nextInt(currIndex);
        }
    }

    /**
     * Initialization.
     *
     * @param dim Integer that is the number of dimensions in the data.
     * @param numDatasets Integer that is the number of datasets.
     * @param numClasses Integer that is the number of classes to generate.
     * @param outDir Directory for the output.
     */
    MultiGaussianLocalityExplorer(int dim, int numDatasets, int numClasses,
            File outDir) {
        this.dim = dim;
        this.numDatatets = numDatasets;
        this.numClasses = numClasses;
        this.outDir = outDir;
    }

    /**
     * This method runs the script for estimating norm correlations with hubness
     * and density in intrinsically high-dimensional Gaussian data.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-outDir", "Path to the output directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-numDatasets", "Number of datasets to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numClasses", "Number of classes to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-dim", "Synthetic data dimensionality.",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File outDir = new File((String) clp.getParamValues("-outDir").get(0));
        int numDatasets = (Integer) clp.getParamValues("-numDatasets").get(0);
        int numClasses = (Integer) clp.getParamValues("-numClasses").get(0);
        int dim = (Integer) clp.getParamValues("-dim").get(0);
        MultiGaussianLocalityExplorer mgle = new MultiGaussianLocalityExplorer(
                dim, numDatasets, numClasses, outDir);
        mgle.runExperiments();
    }
}
