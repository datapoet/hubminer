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
 * A mixture of Gaussians data generator utility class for clustering testing.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class MultiGaussianMixForClusteringTesting {

    private static final int DEFAULT_NUM_CLUSTERS = 5;
    private static final int DEFAULT_NUM_DIMENSIONS = 20;
    private static final int DEFAULT_NUM_INSTANCES = 10000;
    private int numClusters = DEFAULT_NUM_CLUSTERS;
    private int numDimensions = DEFAULT_NUM_DIMENSIONS;
    private int numInstances = DEFAULT_NUM_INSTANCES;
    private boolean usePairedGaussians = true;

    /**
     *
     * @param numClusters Integer that is the number of clusters.
     * @param numDimensions Integer that is the number of dimensions.
     * @param numInstances Integer that is the number of instances.
     * @param usePairedGaussians Boolean flag indicating whether to use paired
     * Gaussian distributions instead of single Gaussians in individual
     * clusters.
     */
    public MultiGaussianMixForClusteringTesting(
            int numClusters,
            int numDimensions,
            int numInstances,
            boolean usePairedGaussians) {
        this.numClusters = numClusters;
        this.numDimensions = numDimensions;
        this.usePairedGaussians = usePairedGaussians;
        this.numInstances = numInstances;
    }

    public DataSet generateRandomCollection() {
        DataSet dset = new DataSet();
        // Set generic attribute names.
        dset.fAttrNames = new String[numDimensions];
        for (int i = 0; i < numDimensions; i++) {
            dset.fAttrNames[i] = "floatAtt" + i;
        }
        dset.data = new ArrayList<>(numInstances);
        // Determine how many instances to pull from which generator.
        float[] generatorProportions = new float[numClusters];
        Random randa = new Random();
        float probTotal = 0;
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            generatorProportions[cIndex] = 1 + randa.nextInt(30);
            probTotal += generatorProportions[cIndex];
        }
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            generatorProportions[cIndex] /= probTotal;
            generatorProportions[cIndex] *= numInstances;
        }
        int[] clusterSizes = new int[numClusters];
        for (int cIndex = 0; cIndex < numClusters; cIndex++) {
            clusterSizes[cIndex] = (int) generatorProportions[cIndex];
        }
        // Initialize data distributions.
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
        for (int i = 0; i < numClusters; i++) {
            for (int j = 0; j < numDimensions; j++) {
                means[i][j] = meanLBound + randa.nextFloat()
                        * (meanUBound - meanLBound);
                stDevs[i][j] = stDevLBound + randa.nextFloat()
                        * (stDevUBound - stDevLBound);
            }
        }
        // Initialize generators.
        DataGenerator[] generators = new DataGenerator[numClusters];
        if (usePairedGaussians) {
            float firstProb;
            for (int i = 0; i < generators.length; i++) {
                firstProb = 0.15f + randa.nextFloat() * 0.7f;
                generators[i] = new PairedSphericGaussians(
                        means[i], stDevs[i], lBounds, uBounds, firstProb);
            }
        } else {
            for (int i = 0; i < generators.length; i++) {
                generators[i] = new MultiDimensionalSphericGaussianGenerator(
                        means[i], stDevs[i], lBounds, uBounds);
            }
        }
        MixtureOfFloatGenerators workingGenerator =
                new MixtureOfFloatGenerators(generators);
        // The labels are automatically set inside this call.
        DataInstance[] instances = workingGenerator.generateDataInstances(
                clusterSizes);
        for (int i = 0; i < instances.length; i++) {
            dset.addDataInstance(instances[i]);
            instances[i].embedInDataset(dset);
        }
        return dset;
    }

    /**
     * Generate the data and persist.
     *
     * @param outFile File where to persist the data to.
     * @throws Exception
     */
    public void generateAndWriteToFile(File outFile) throws Exception {
        DataSet dset = generateRandomCollection();
        IOARFF persister = new IOARFF();
        persister.save(dset, outFile.getPath(), null);
    }

    /**
     * Generates and saves the data.
     *
     * @param args Command line args, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-numClusters", "Number of clusters to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numDimensions", "Number of dimensions to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-numInstances", "Number of instances to generate.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-outFile", "Output arff file.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-paired_gaussian", "true/false on using paired gaussians "
                + "as generators.", CommandLineParser.BOOLEAN, false, false);
        clp.parseLine(args);
        MultiGaussianMixForClusteringTesting worker =
                new MultiGaussianMixForClusteringTesting(
                (Integer) clp.getParamValues("-numClusters").get(0),
                (Integer) clp.getParamValues("-numDimensions").get(0),
                (Integer) clp.getParamValues("-numInstances").get(0),
                (Boolean) clp.getParamValues("-paired_gaussian").get(0));
        worker.generateAndWriteToFile(new File((String) clp.getParamValues(
                "-outFile").get(0)));

    }
}
