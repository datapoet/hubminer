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

import data.generators.DataGenerator;
import data.generators.GaussianMislabeledDataGenerator;
import data.generators.MixtureOfFloatGenerators;
import data.generators.MultiDimensionalSphericGaussianGenerator;
import data.generators.PairedSphericGaussians;
import data.representation.DataInstance;
import data.representation.DataSet;
import ioformat.IOARFF;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;
import util.CommandLineParser;

/**
 * A utility class for generating some overlapping data with label noise.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MislabeledInterwinedDataGenerator {

    private int numClusters = 10;
    private int numDimensions = 25;
    private boolean usePairedGaussians = true;
    float probDependence = 0.85f;
    // Meta-generator for generating synthetic data points.
    MixtureOfFloatGenerators workingGenerator;

    /**
     *
     * @param numClusters Integer that is the number of clusters.
     * @param numDimensions Integer that is the number of dimensions to
     * generate.
     * @param usePairedGaussians Boolean indicating whether to use paired
     * Gaussian distributions for data clusters.
     */
    public MislabeledInterwinedDataGenerator(
            int numClusters, int numDimensions, boolean usePairedGaussians) {
        this.numClusters = numClusters;
        this.numDimensions = numDimensions;
        this.usePairedGaussians = usePairedGaussians;
    }

    /**
     * Initialize data generators.
     */
    private void initializeGenerators() {
        Random randa = new Random();
        // Parameter initialization.
        float[][] means = new float[numClusters][numDimensions];
        float[][] stDevs = new float[numClusters][numDimensions];
        float[] lBounds = new float[numDimensions];
        float[] uBounds = new float[numDimensions];
        for (int i = 0; i < numDimensions; i++) {
            lBounds[i] = -Float.MAX_VALUE;
            uBounds[i] = Float.MAX_VALUE;
        }
        float meanLBound = - 20;
        float meanUBound = 20;
        float stDevLBound = 2;
        float stDevUBound = 5;
        int previous;
        float choice;
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            if (cIndex > 0) {
                for (int d = 0; d < numDimensions; d++) {
                    choice = randa.nextFloat();
                    if (choice < probDependence) {
                        // Induce some overlap.
                        previous = randa.nextInt(cIndex);
                        means[cIndex][d] = means[previous][d];
                        stDevs[cIndex][d] = stDevLBound + randa.nextFloat()
                                * (stDevUBound - stDevLBound);
                    } else {
                        means[cIndex][d] = meanLBound + randa.nextFloat()
                                * (meanUBound - meanLBound);
                        stDevs[cIndex][d] = stDevLBound + randa.nextFloat()
                                * (stDevUBound - stDevLBound);
                    }
                }
            } else {
                for (int j = 0; j < numDimensions; j++) {
                    means[cIndex][j] = meanLBound + randa.nextFloat()
                            * (meanUBound - meanLBound);
                    stDevs[cIndex][j] = stDevLBound + randa.nextFloat()
                            * (stDevUBound - stDevLBound);
                }
            }
        }
        // Initialize the generator objects themselves.
        DataGenerator[] generators = new DataGenerator[numClusters];
        if (usePairedGaussians) {
            float firstProb;
            for (int cIndex = 0; cIndex < generators.length; cIndex++) {
                firstProb = 0.15f + randa.nextFloat() * 0.7f;
                generators[cIndex] = new PairedSphericGaussians(
                        means[cIndex], stDevs[cIndex], lBounds, uBounds,
                        firstProb);
            }
        } else {
            for (int cIndex = 0; cIndex < generators.length; cIndex++) {
                generators[cIndex] =
                        new MultiDimensionalSphericGaussianGenerator(
                        means[cIndex], stDevs[cIndex], lBounds,
                        uBounds);
            }
        }
        workingGenerator = new MixtureOfFloatGenerators(generators);
    }

    private DataSet generateRandomData(int numInstances, float mislabeledProb) {
        DataSet dset = new DataSet();
        dset.fAttrNames = new String[numDimensions];
        dset.data = new ArrayList<>(numInstances);
        // Determine how many instances to generate from which generator.
        float[] generatorProportions = new float[numClusters];
        Random randa = new Random();
        float probTotal = 0;
        for (int i = 0; i < numClusters; i++) {
            generatorProportions[i] = 1 + randa.nextInt(30);
            probTotal += generatorProportions[i];
        }
        for (int i = 0; i < numClusters; i++) {
            generatorProportions[i] /= probTotal;
            generatorProportions[i] *= numInstances;
        }
        // Truncate to obtain cluster sizes.
        int[] clusterSizes = new int[numClusters];
        for (int i = 0; i < numClusters; i++) {
            clusterSizes[i] = (int) generatorProportions[i];
        }
        // Now generate random data.
        DataInstance[] instances =
                workingGenerator.generateDataInstances(clusterSizes);
        float choice;
        int changedLabel;
        // Induce some label noise.
        for (int i = 0; i < instances.length; i++) {
            dset.addDataInstance(instances[i]);
            instances[i].embedInDataset(dset);
            choice = randa.nextFloat();
            if (choice < mislabeledProb) {
                do {
                    changedLabel = randa.nextInt(numClusters);
                } while (changedLabel == instances[i].getCategory());
                instances[i].setCategory(changedLabel);
            }
        }
        return dset;
    }

    /**
     * Generate the data and write it to a file.
     *
     * @param numInstances Integer that is the number of instances.
     * @param outFile Output file.
     * @param mislabeledProb Probability of randomly flipping a label.
     * @throws Exception
     */
    public void generateAndWriteToFile(int numInstances, File outFile,
            float mislabeledProb) throws Exception {
        DataSet dset = generateRandomData(numInstances, mislabeledProb);
        IOARFF persister = new IOARFF();
        persister.save(dset, outFile.getPath(), null);
    }

    /**
     * Generates the data and persists it to a file.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-numDimensions", "Number of dimensions to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numInstances", "Number of instances to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numCat", "Number of categories to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-flipProb", "Mislabeling probability.",
                CommandLineParser.FLOAT, true, false);
        clp.addParam("-outFile", "Output arff file path.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-paired_gaussian", "True/false. Whether to use paired "
                + "Gaussian distributions.", CommandLineParser.BOOLEAN, false,
                false);
        clp.parseLine(args);
        GaussianMislabeledDataGenerator worker =
                new GaussianMislabeledDataGenerator(
                (Integer) clp.getParamValues("-numCat").get(0),
                (Integer) clp.getParamValues("-numDimensions").get(0),
                (Boolean) clp.getParamValues("-paired_gaussian").get(0));
        worker.initializeGenerators();
        worker.generateAndWriteToFile(
                (Integer) clp.getParamValues("-numInstances").get(0),
                new File((String) clp.getParamValues("-outFile").get(0)),
                (Float) clp.getParamValues("-flipProb").get(0));
    }
}
