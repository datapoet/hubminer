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
package data.generators.util;

import data.representation.DataInstance;
import data.representation.DataSet;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import util.CommandLineParser;

/**
 * This utility class generates overlapping Gaussian test data. It has an
 * imbalanced mode to use for testing under class imbalance.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class OverlappingGaussianGenerator {

    private float[][] stDevs;
    private float[][] means;
    private int numDS;
    private int dim;
    private int numCat;
    private DataSet dset;
    private File outDir;
    private File[] dataDirs;
    private static int catMin = 20;
    private static int catMax = 1000;
    private boolean imbalancedMode = false;

    public OverlappingGaussianGenerator() {
    }

    /**
     *
     * @param imbalancedMode Boolean flag indicating whether to generate class
     * imbalanced data.
     */
    public OverlappingGaussianGenerator(boolean imbalancedMode) {
        this.imbalancedMode = imbalancedMode;
    }

    /**
     * Generates the data and persists the data and the generated distribution
     * information to a file.
     *
     * @throws Exception
     */
    private void generateAndPersist() throws Exception {
        Random randa = new Random();
        // Generates a sequence of datasets.
        dataDirs = new File[numDS];
        for (int i = 0; i < numDS; i++) {
            dataDirs[i] = new File(outDir, "ds" + i);
        }
        // First generate all data distributions.
        float alpha = 0.75f;
        float beta = 1.5f;
        float choice;
        float compensator = 0.5f;
        for (int i = 0; i < numDS; i++) {
            stDevs = new float[numCat][dim];
            means = new float[numCat][dim];
            for (int d = 0; d < dim; d++) {
                means[0][d] = 0;
                stDevs[0][d] = (float) Math.max(0.1, 4 + randa.nextGaussian());
            }
            for (int catIndex = 1; catIndex < numCat; catIndex++) {
                // Choose an already existing class distribution to overlap with
                int brother = chooseBrotherDistribution(catIndex, randa);
                // Update the minority and majority counts.
                for (int d = 0; d < dim; d++) {
                    // One way or the other, 50-50 chance.
                    choice = randa.nextFloat();
                    if (choice < 0.5) {
                        means[catIndex][d] = means[brother][d]
                                + compensator * stDevs[brother][d];
                    } else {
                        means[catIndex][d] = means[brother][d]
                                - compensator * stDevs[brother][d];
                    }
                    stDevs[catIndex][d] = alpha * stDevs[brother][d]
                            + (beta - alpha) * randa.nextFloat()
                            * stDevs[brother][d];
                }
            }
            // Persist the generated model.
            File outDistrFile = new File(dataDirs[i], "generatingModel.txt");
            FileUtil.createFile(outDistrFile);
            PrintWriter pw = new PrintWriter(new FileWriter(outDistrFile));
            try {
                for (int c = 0; c < numCat; c++) {
                    pw.println("category: " + c);
                    pw.print("means: " + means[c][0]);
                    for (int j = 1; j < dim; j++) {
                        pw.print("," + means[c][j]);
                    }
                    pw.println();
                    pw.print("stDevs: " + stDevs[c][0]);
                    for (int j = 1; j < dim; j++) {
                        pw.print("," + stDevs[c][j]);
                    }
                    pw.println();
                }
            } catch (Exception e) {
                System.err.println(e.getMessage());
                throw e;
            } finally {
                pw.close();
            }
            // Generate the data from the selected distributions.
            int size;
            DataInstance instance;
            dset = new DataSet();
            // Create generic attribute names.
            dset.fAttrNames = new String[dim];
            for (int j = 0; j < dim; j++) {
                dset.fAttrNames[j] = "att" + j;
            }
            dset.data = new ArrayList<>(numCat * catMax);
            // Probability of a class being a majority class. Less than 0.5.
            float majorityProb =
                    Math.min(1 + randa.nextInt(numCat / 2), numCat / 2);
            majorityProb /= numCat;
            for (int catIndex = 0; catIndex < numCat; catIndex++) {
                if (!imbalancedMode) {
                    size = catMin + (int) ((catMax - catMin)
                            * randa.nextFloat());
                } else {
                    float threshold = majorityProb;
                    choice = randa.nextFloat();
                    if (choice < threshold) {
                        // Majority mode.
                        size = catMax - (int) ((catMax - catMin)
                                * randa.nextFloat() * 0.3);
                    } else {
                        // Minority mode.
                        size = catMin + (int) ((catMax - catMin)
                                * randa.nextFloat() * 0.15);
                    }
                }

                System.out.println("cSize " + catIndex + ":" + size);
                for (int j = 0; j < size; j++) {
                    instance = new DataInstance(dset);
                    dset.addDataInstance(instance);
                    instance.embedInDataset(dset);
                    instance.setCategory(catIndex);
                    for (int d = 0; d < dim; d++) {
                        instance.fAttr[d] =
                                means[catIndex][d] + stDevs[catIndex][d]
                                * (float) randa.nextGaussian();
                    }
                }
            }
            System.out.println("data size: " + dset.size());
            // Persist the data.
            IOARFF persister = new IOARFF();
            persister.saveLabeledWithIdentifiers(dset,
                    (new File(dataDirs[i], "dataset" + i + ".arff")).getPath(),
                    null);
            System.out.println("finished dataset " + i);
        }
    }

    /**
     * Generates the data without persisting the model or the data to a file.
     *
     * @param dim Integer that is the number of dimensions.
     * @param numCat Integer that is the number of categories.
     * @param imbalancedMode Boolean flag indicating whether to generate an
     * imbalanced dataset.
     * @param catMin Integer that is the minimal category size.
     * @param catMax Integer that is the maximal category size.
     * @return DataSet that was generated.
     * @throws Exception
     */
    public static DataSet generate(int dim, int numCat,
            boolean imbalancedMode, int catMin, int catMax) throws Exception {
        Random randa = new Random();
        // First generate all data distributions.
        float alpha = 0.75f;
        float beta = 1.5f;
        float choice;
        float compensator = 0.5f;
        float[][] stDevs = new float[numCat][dim];
        float[][] means = new float[numCat][dim];
        for (int d = 0; d < dim; d++) {
            means[0][d] = 0;
            stDevs[0][d] = (float) Math.max(0.1, 4 + randa.nextGaussian());
        }
        for (int catIndex = 1; catIndex < numCat; catIndex++) {
            // Choose an already existing class distribution to overlap with
            int brother = chooseBrotherDistribution(catIndex, randa);
            // Update the minority and majority counts.
            for (int d = 0; d < dim; d++) {
                // One way or the other, 50-50 chance.
                choice = randa.nextFloat();
                if (choice < 0.5) {
                    means[catIndex][d] = means[brother][d]
                            + compensator * stDevs[brother][d];
                } else {
                    means[catIndex][d] = means[brother][d]
                            - compensator * stDevs[brother][d];
                }
                stDevs[catIndex][d] = alpha * stDevs[brother][d]
                        + (beta - alpha) * randa.nextFloat()
                        * stDevs[brother][d];
            }
        }
        // Generate the data from the selected distributions.
        int size;
        DataInstance instance;
        DataSet dset = new DataSet();
        // Create generic attribute names.
        dset.fAttrNames = new String[dim];
        for (int j = 0; j < dim; j++) {
            dset.fAttrNames[j] = "att" + j;
        }
        dset.data = new ArrayList<>(numCat * catMax);
        // Probability of a class being a majority class. Less than 0.5.
        float majorityProb =
                Math.min(1 + randa.nextInt(numCat / 2), numCat / 2);
        majorityProb /= numCat;
        for (int catIndex = 0; catIndex < numCat; catIndex++) {
            if (!imbalancedMode) {
                size = catMin + (int) ((catMax - catMin)
                        * randa.nextFloat());
            } else {
                float threshold = majorityProb;
                choice = randa.nextFloat();
                if (choice < threshold) {
                    // Majority mode.
                    size = catMax - (int) ((catMax - catMin)
                            * randa.nextFloat() * 0.3);
                } else {
                    // Minority mode.
                    size = catMin + (int) ((catMax - catMin)
                            * randa.nextFloat() * 0.15);
                }
            }
            for (int j = 0; j < size; j++) {
                instance = new DataInstance(dset);
                dset.addDataInstance(instance);
                instance.embedInDataset(dset);
                instance.setCategory(catIndex);
                for (int d = 0; d < dim; d++) {
                    instance.fAttr[d] =
                            means[catIndex][d] + stDevs[catIndex][d]
                            * (float) randa.nextGaussian();
                }
            }
        }
        return dset;
    }

    /**
     * Choose a distribution to overlap with.
     *
     * @param currIndex Current index to couple.
     * @param randa Random generator to use.
     * @return Integer that is the index of a previously generated distribution
     * to couple this one with.
     */
    private static int chooseBrotherDistribution(int currIndex, Random randa) {
        if (currIndex == 1) {
            return 0;
        } else {
            return randa.nextInt(currIndex);
        }
    }

    /**
     *
     * @param dim Integer that is the number of features.
     * @param numDS Integer that is the number of data sets to generate.
     * @param numCat Integer that is the number of categories per dataset.
     * @param outDir String that is the path to the output directory.
     * @param imbalancedMode Boolean flag indicating whether to use the class
     * imbalanced data generation mode.
     */
    OverlappingGaussianGenerator(
            int dim,
            int numDS,
            int numCat,
            File outDir,
            boolean imbalancedMode) {
        this.dim = dim;
        this.numDS = numDS;
        this.numCat = numCat;
        this.outDir = outDir;
        this.imbalancedMode = imbalancedMode;
    }

    /**
     * Generates the data according to user input.
     *
     * @param args Command line args, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-numDimensions", "Number of dimensions to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numDatasets", "Number of datasets to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numCat", "Number of categories to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-outDir", "Output directory for generated files.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-imbalanced", "True/false. Whether to "
                + "generate imbalanced data.", CommandLineParser.BOOLEAN, false,
                false);
        clp.parseLine(args);
        OverlappingGaussianGenerator ogg = new OverlappingGaussianGenerator(
                (Integer) clp.getParamValues("-numDimensions").get(0),
                (Integer) clp.getParamValues("-numDatasets").get(0),
                (Integer) clp.getParamValues("-numCat").get(0),
                new File((String) (clp.getParamValues("-outDir").get(0))),
                (Boolean) clp.getParamValues("-imbalanced").get(0));
        ogg.generateAndPersist();
    }
}
