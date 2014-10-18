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
package data.generators;

import data.generators.util.MultiGaussianMixForClusteringTesting;
import data.representation.DataInstance;
import data.representation.DataSet;
import ioformat.IOARFF;
import java.io.File;
import java.util.ArrayList;
import java.util.Random;

/**
 * This class generates a noisy Gaussian mix of points where Gaussian
 * distributions are places in a region of uniform noise.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class NoisyGaussianMix {

    private int numClusters = 10;
    private int numDimensions = 50;
    private int numInstances = 10000;
    private int numNoisyInstances = 2000;
    private boolean usePairedGaussians = true;

    /**
     *
     * @param numClusters Number of clusters to generate.
     * @param numDimensions Number of float features.
     * @param numInstances Number of instances.
     * @param usePairedGaussians True if using paired Gaussian distributions.
     * @param numNoisyInstances Number of noisy instances.
     */
    public NoisyGaussianMix(
            int numClusters,
            int numDimensions,
            int numInstances,
            boolean usePairedGaussians,
            int numNoisyInstances) {
        this.numClusters = numClusters;
        this.numDimensions = numDimensions;
        this.usePairedGaussians = usePairedGaussians;
        this.numInstances = numInstances;
        this.numNoisyInstances = numNoisyInstances;
    }

    /**
     * Introduce noise to the dataset.
     *
     * @param dset DataSet object.
     * @param moreNoisyInstances Number of noisy instances to add.
     */
    public void addNoiseToCollection(DataSet dset, int numNewNoisyInstances) {
        float[] lBounds = new float[numDimensions];
        float[] uBounds = new float[numDimensions];
        for (int i = 0; i < numDimensions; i++) {
            lBounds[i] = -200;
            uBounds[i] = 200;
        }
        UniformGenerator noiseGenerator =
                new UniformGenerator(lBounds, uBounds);
        DataInstance[] instances =
                noiseGenerator.generateDataInstances(numNewNoisyInstances);
        for (int i = 0; i < instances.length; i++) {
            dset.addDataInstance(instances[i]);
            instances[i].embedInDataset(dset);
            // Noisy instance labels are marked as -1.
            instances[i].setCategory(-1);
        }
    }

    /**
     * Generate the random data set.
     *
     * @return
     */
    public DataSet generateRandomDataSet() {
        DataSet dset = new DataSet();
        dset.fAttrNames = new String[numDimensions];
        for (int i = 0; i < numDimensions; i++) {
            dset.fAttrNames[i] = "floatAtt" + i;
        }
        dset.data = new ArrayList<>(numInstances + numNoisyInstances);
        // Calculate how many instances to generate by which generator.
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
        int[] clusterSizes = new int[numClusters];
        for (int i = 0; i < numClusters; i++) {
            clusterSizes[i] = (int) generatorProportions[i];
        }
        // Initialization of the specs.
        float[][] means = new float[numClusters][numDimensions];
        float[][] stDevs = new float[numClusters][numDimensions];
        float[] lBounds = new float[numDimensions];
        float[] uBounds = new float[numDimensions];
        for (int i = 0; i < numDimensions; i++) {
            lBounds[i] = -200000;
            uBounds[i] = 200000;
        }
        float meanLBound = - 150;
        float meanUBound = 150;
        float stDevLBound = 10;
        float stDevUBound = 60;
        for (int i = 0; i < numClusters; i++) {
            for (int j = 0; j < numDimensions; j++) {
                means[i][j] = meanLBound + randa.nextFloat()
                        * (meanUBound - meanLBound);
                stDevs[i][j] = stDevLBound + randa.nextFloat()
                        * (stDevUBound - stDevLBound);
            }
        }
        // Initialization of the generators.
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
        DataInstance[] instances = workingGenerator.generateDataInstances(
                clusterSizes);
        for (int i = 0; i < instances.length; i++) {
            dset.addDataInstance(instances[i]);
            instances[i].embedInDataset(dset);
        }
        UniformGenerator noiseGenerator =
                new UniformGenerator(lBounds, uBounds);
        instances = noiseGenerator.generateDataInstances(numNoisyInstances);
        for (int i = 0; i < instances.length; i++) {
            dset.addDataInstance(instances[i]);
            instances[i].embedInDataset(dset);
            // Noisy instance labels are marked as -1.
            instances[i].setCategory(-1);
        }
        return dset;
    }

    /**
     * Generate the random data and persist to a file.
     *
     * @param outFile File to persist the data to.
     * @throws Exception
     */
    public void generateAndWriteToFile(File outFile) throws Exception {
        DataSet dc = generateRandomDataSet();
        IOARFF persister = new IOARFF();
        persister.save(dc, outFile.getPath(), null);
    }

    /**
     * Command line specs.
     */
    public static void info() {
        System.out.println("4 args...");
        System.out.println("arg0: numClusters");
        System.out.println("arg1: numDimensions");
        System.out.println("arg2: numInstances");
        System.out.println("arg3: true/false on using paired gaussians as"
                + "generators");
        System.out.println("arg4: output arff file");
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 5) {
            info();
        } else {
            MultiGaussianMixForClusteringTesting worker =
                    new MultiGaussianMixForClusteringTesting(
                    Integer.parseInt(args[0]),
                    Integer.parseInt(args[1]),
                    Integer.parseInt(args[2]),
                    Boolean.parseBoolean(args[3]));
            worker.generateAndWriteToFile(new File(args[4]));
        }
    }
}
