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

import data.generators.MultiDimensionalSphericGaussianGenerator;
import data.neighbors.NeighborSetFinder;
import data.representation.DataInstance;
import data.representation.DataSet;
import distances.concentration.ConcentrationCalculator;
import distances.primary.CombinedMetric;
import distances.primary.MinkowskiMetric;
import feature.correlation.DistanceCorrelation;
import feature.correlation.PearsonCorrelation;
import ioformat.FileUtil;
import ioformat.IOARFF;
import java.io.File;
import java.util.ArrayList;

import java.util.Arrays;
import java.util.Random;
import util.CommandLineParser;

/**
 * This class implements an experiment for tracking hub localization in
 * synthetic Gaussian intrinsically high-dimensional data, incrementally. The
 * k-nearest neighbor sets are updated after every new synthetic data instance
 * insertion. The localization is tracked over several different neighborhood
 * sizes and for several metrics - Euclidean, Manhattan and Fractional.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class GaussianHubnessLocalizer {

    // Directory for the results.
    private File outDir;
    // Dimensionality of the synthetic data.
    private int dim;
    // Minimal number of instances (when to start the output).
    private int minInst;
    // Maximal number of instances (when to finish the output).
    private int maxInst;
    // The maximal neighborhood size to check.
    private int maxK;
    // The distance matrices for Manhattan, Euclidean and Fractional distances.
    private float[][] distancesMan;
    private float[][] distancesEuc;
    private float[][] distancesFrac;
    // The Manhattan distance auxiliary kNN arrays.
    private int[][] kneighborsMan = null;
    private float[][] kdistancesMan = null;
    private int[] kcurrLenMan = null;
    private float[] kneighborFrequenciesMan = null;
    // The Euclidean distance auxiliary kNN arrays.
    private int[][] kneighborsEuc = null;
    private float[][] kdistancesEuc = null;
    private int[] kcurrLenEuc = null;
    private float[] kneighborFrequenciesEuc = null;
    // The Fractional distance auxiliary kNN arrays.
    private int[][] kneighborsFrac = null;
    private float[][] kdistancesFrac = null;
    private int[] kcurrLenFrac = null;
    private float[] kneighborFrequenciesFrac = null;
    // Cluster medoids for each distance type.
    private DataInstance[] medoidsMan;
    private DataInstance[] medoidsEuc;
    private DataInstance[] medoidsFrac;

    /**
     * Initialization.
     *
     * @param outDir Directory for the output.
     * @param dim Integer that is the desired data dimensionality.
     * @param minInst Integer that is the minimal number of instances.
     * @param maxInst Integer that is the maximal number of instances.
     * @param maxK Integer that is the maximal neighborhood size to consider.
     */
    public GaussianHubnessLocalizer(File outDir, int dim, int minInst,
            int maxInst, int maxK) {
        this.outDir = outDir;
        this.dim = dim;
        this.minInst = minInst;
        this.maxInst = maxInst;
        this.maxK = maxK;
    }

    /**
     * This method runs the entire experiment, incrementally inserting instances
     * into the dataset and updating the kNN sets and tracking for medoid
     * localization in the clusters.
     *
     * @throws Exception
     */
    public void runHubnessLocalizationExperiment() throws Exception {
        // Initialize the distance calculation objects.
        CombinedMetric cmetMan = CombinedMetric.FLOAT_MANHATTAN;
        CombinedMetric cmetEuc = CombinedMetric.FLOAT_EUCLIDEAN;
        CombinedMetric cmetFrac =
                new CombinedMetric(null, new MinkowskiMetric(0.5f),
                CombinedMetric.DEFAULT);
        // Generate the synthetic dataset.
        Random randa = new Random();
        // Initialize the means and the standard deviations.
        float[] featureMeans = new float[dim];
        Arrays.fill(featureMeans, 0);
        float[] featureStDevs = new float[dim];
        for (int dIndex = 0; dIndex < dim; dIndex++) {
            featureStDevs[dIndex] = randa.nextFloat();
        }
        // Set the value bounds.
        float[] lBounds = new float[dim];
        float[] uBounds = new float[dim];
        Arrays.fill(lBounds, -10);
        Arrays.fill(uBounds, 10);
        // Initialize the Gaussian generator.
        MultiDimensionalSphericGaussianGenerator gen =
                new MultiDimensionalSphericGaussianGenerator(
                featureMeans, featureStDevs, lBounds, uBounds);
        // Initialize the dataset.
        DataSet dset = new DataSet();
        dset.fAttrNames = new String[dim];
        for (int dIndex = 0; dIndex < dim; dIndex++) {
            dset.fAttrNames[dIndex] = "f" + dIndex;
        }
        dset.data = new ArrayList<>(maxInst);
        // Generate all the data instances.
        DataInstance instance;
        for (int i = 0; i < maxInst; i++) {
            instance = new DataInstance(dset);
            instance.fAttr = gen.generateFloat();
            dset.addDataInstance(instance);
        }
        // Persist the generated experimental data.
        File outDsetFile = new File(outDir, "data.arff");
        IOARFF pers = new IOARFF();
        pers.saveLabeledWithIdentifiers(dset, outDsetFile.getPath(), null);
        // Calculate the distance matrices in a multi-threaded way.
        distancesMan = dset.calculateDistMatrixMultThr(cmetMan, 4);
        distancesEuc = dset.calculateDistMatrixMultThr(cmetEuc, 4);
        distancesFrac = dset.calculateDistMatrixMultThr(cmetFrac, 4);
        // Notify the user about the end of distance calculations.
        System.out.println("All distances calculated.");
        // Initialize the result sets. Results are also represented as DataSet
        // objects.
        DataSet resultsManh = new DataSet();
        resultsManh.fAttrNames = new String[2 + 10 * maxK];
        // Relative contrast and relative variance do not depend on neighborhood
        // size.
        resultsManh.fAttrNames[0] = "relativeContrast";
        resultsManh.fAttrNames[1] = "relativeVariance";
        // The remaining measures depend on neighborhood size.
        for (int kIndex = 0; kIndex < maxK; kIndex++) {
            // Ratio between the hub and medoid distance.
            resultsManh.fAttrNames[2 + 10 * kIndex] =
                    "hDist/mDist_ratio" + (kIndex + 1);
            // Hub to medoid distance.
            resultsManh.fAttrNames[3 + 10 * kIndex] = "hmDist" + (kIndex + 1);
            // Normalized hub to medoid distance.
            resultsManh.fAttrNames[4 + 10 * kIndex] =
                    "hmDist/avgDist_ratio" + (kIndex + 1);
            // Hub distance.
            resultsManh.fAttrNames[5 + 10 * kIndex] = "hDist" + (kIndex + 1);
            // Correlation between hubness and norm.
            resultsManh.fAttrNames[6 + 10 * kIndex] =
                    "normHubnessCorr" + (kIndex + 1);
            // Correlation between hubness and density.
            resultsManh.fAttrNames[7 + 10 * kIndex] =
                    "densityHubnessCorr" + (kIndex + 1);
            // Correlation between norm and density.
            resultsManh.fAttrNames[8 + 10 * kIndex] =
                    "densityNormCorr" + (kIndex + 1);
            // Distance correlation between hubness and norm.
            resultsManh.fAttrNames[9 + 10 * kIndex] =
                    "normHubnessDistCorr" + (kIndex + 1);
            // Distance correlation between hubness and density.
            resultsManh.fAttrNames[10 + 10 * kIndex] =
                    "densityHubnessDistCorr" + (kIndex + 1);
            // Distance correlation between density and norm.
            resultsManh.fAttrNames[11 + 10 * kIndex] =
                    "densityNormDistCorr" + (kIndex + 1);
        }
        // Initialize the result holder instances.
        resultsManh.data = new ArrayList<>(maxInst - minInst);
        for (int i = minInst; i < maxInst; i++) {
            instance = new DataInstance(resultsManh);
            Arrays.fill(instance.fAttr, 0);
            resultsManh.addDataInstance(instance);
        }
        // Now the Euclidean distance.
        DataSet resultsEuc = new DataSet();
        resultsEuc.fAttrNames = new String[2 + 10 * maxK];
        // Relative contrast and relative variance do not depend on neighborhood
        // size.
        resultsEuc.fAttrNames[0] = "relativeContrast";
        resultsEuc.fAttrNames[1] = "relativeVariance";
        for (int kIndex = 0; kIndex < maxK; kIndex++) {
            // Ratio between the hub and medoid distance.
            resultsEuc.fAttrNames[2 + 10 * kIndex] =
                    "hDist/mDist_ratio" + (kIndex + 1);
            // Hub to medoid distance.
            resultsEuc.fAttrNames[3 + 10 * kIndex] = "hmDist" + (kIndex + 1);
            // Normalized hub to medoid distance.
            resultsEuc.fAttrNames[4 + 10 * kIndex] =
                    "hmDist/avgDist_ratio" + (kIndex + 1);
            // Hub distance.
            resultsEuc.fAttrNames[5 + 10 * kIndex] = "hDist" + (kIndex + 1);
            // Correlation between hubness and norm.
            resultsEuc.fAttrNames[6 + 10 * kIndex] =
                    "normHubnessCorr" + (kIndex + 1);
            // Correlation between hubness and density.
            resultsEuc.fAttrNames[7 + 10 * kIndex] =
                    "densityHubnessCorr" + (kIndex + 1);
            // Correlation between norm and density.
            resultsEuc.fAttrNames[8 + 10 * kIndex] =
                    "densityNormCorr" + (kIndex + 1);
            // Distance correlation between hubness and norm.
            resultsEuc.fAttrNames[9 + 10 * kIndex] =
                    "normHubnessDistCorr" + (kIndex + 1);
            // Distance correlation between hubness and density.
            resultsEuc.fAttrNames[10 + 10 * kIndex] =
                    "densityHubnessDistCorr" + (kIndex + 1);
            // Distance correlation between density and norm.
            resultsEuc.fAttrNames[11 + 10 * kIndex] =
                    "densityNormDistCorr" + (kIndex + 1);
        }
        resultsEuc.data = new ArrayList<>(maxInst - minInst);
        for (int i = minInst; i < maxInst; i++) {
            instance = new DataInstance(resultsEuc);
            Arrays.fill(instance.fAttr, 0);
            resultsEuc.addDataInstance(instance);
        }
        // Now the fractional distances.
        DataSet resultsFrac = new DataSet();
        resultsFrac.fAttrNames = new String[2 + 10 * maxK];
        // Relative contrast and relative variance do not depend on neighborhood
        // size.
        resultsFrac.fAttrNames[0] = "relativeContrast";
        resultsFrac.fAttrNames[1] = "relativeVariance";
        for (int kIndex = 0; kIndex < maxK; kIndex++) {
            // Ratio between the hub and medoid distance.
            resultsFrac.fAttrNames[2 + 10 * kIndex] =
                    "hDist/mDist_ratio" + (kIndex + 1);
            // Hub to medoid distance.
            resultsFrac.fAttrNames[3 + 10 * kIndex] = "hmDist" + (kIndex + 1);
            // Normalized hub to medoid distance.
            resultsFrac.fAttrNames[4 + 10 * kIndex] =
                    "hmDist/avgDist_ratio" + (kIndex + 1);
            // Hub distance.
            resultsFrac.fAttrNames[5 + 10 * kIndex] = "hDist" + (kIndex + 1);
            // Correlation between hubness and norm.
            resultsFrac.fAttrNames[6 + 10 * kIndex] =
                    "normHubnessCorr" + (kIndex + 1);
            // Correlation between hubness and density.
            resultsFrac.fAttrNames[7 + 10 * kIndex] =
                    "densityHubnessCorr" + (kIndex + 1);
            // Correlation between norm and density.
            resultsFrac.fAttrNames[8 + 10 * kIndex] =
                    "densityNormCorr" + (kIndex + 1);
            // Distance correlation between hubness and norm.
            resultsFrac.fAttrNames[9 + 10 * kIndex] =
                    "normHubnessDistCorr" + (kIndex + 1);
            // Distance correlation between hubness and density.
            resultsFrac.fAttrNames[10 + 10 * kIndex] =
                    "densityHubnessDistCorr" + (kIndex + 1);
            // Distance correlation between density and norm.
            resultsFrac.fAttrNames[11 + 10 * kIndex] =
                    "densityNormDistCorr" + (kIndex + 1);
        }
        resultsFrac.data = new ArrayList<>(maxInst - minInst);
        for (int i = minInst; i < maxInst; i++) {
            instance = new DataInstance(resultsFrac);
            Arrays.fill(instance.fAttr, 0);
            resultsFrac.addDataInstance(instance);
        }
        // Initialize the zero-centered centroid.
        DataInstance centroid = new DataInstance(dset);
        Arrays.fill(centroid.fAttr, 0);
        // Initialize all the arrays.
        medoidsMan = new DataInstance[maxInst];
        medoidsEuc = new DataInstance[maxInst];
        medoidsFrac = new DataInstance[maxInst];
        kneighborsMan = new int[dset.size()][maxK];
        kdistancesMan = new float[dset.size()][maxK];
        kcurrLenMan = new int[dset.size()];
        kneighborsEuc = new int[dset.size()][maxK];
        kdistancesEuc = new float[dset.size()][maxK];
        kcurrLenEuc = new int[dset.size()];
        kneighborsFrac = new int[dset.size()][maxK];
        kdistancesFrac = new float[dset.size()][maxK];
        kcurrLenFrac = new int[dset.size()];
        // Initialize the norms and the densities.
        double[] normsMan = new double[maxInst];
        double[] normsEuc = new double[maxInst];
        double[] normsFrac = new double[maxInst];

        double[] densitiesMan;
        double[] densitiesEuc;
        double[] densitiesFrac;
        // Calculate all the data instance norms.
        for (int i = 0; i < maxInst; i++) {
            instance = dset.data.get(i);
            normsMan[i] = cmetMan.dist(instance, centroid);
            normsEuc[i] = cmetEuc.dist(instance, centroid);
            normsFrac[i] = cmetFrac.dist(instance, centroid);
        }
        // First generate partial stats until minInst is reached.
        for (int index = 1; index < minInst; index++) {
            medoidsMan[index] =
                    dset.getMedoidUpUntilIdex(centroid, cmetMan, index + 1);
            medoidsEuc[index] =
                    dset.getMedoidUpUntilIdex(centroid, cmetEuc, index + 1);
            medoidsFrac[index] =
                    dset.getMedoidUpUntilIdex(centroid, cmetFrac, index + 1);

            int l;
            for (int i = 0; i < index; i++) {
                int j = index - i - 1;
                int other = index;
                if (kcurrLenMan[i] > 0) {
                    if (kcurrLenMan[i] == maxK) {
                        if (distancesMan[i][j] < kdistancesMan[i][
                                kcurrLenMan[i] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesMan[i][j]
                                    < kdistancesMan[i][l - 1]) {
                                kdistancesMan[i][l] = kdistancesMan[i][l - 1];
                                kneighborsMan[i][l] = kneighborsMan[i][l - 1];
                                l--;
                            }
                            kdistancesMan[i][l] = distancesMan[i][j];
                            kneighborsMan[i][l] = i + j + 1;
                        }
                    } else {
                        if (distancesMan[i][j]
                                < kdistancesMan[i][kcurrLenMan[i] - 1]) {
                            // Search and insert.
                            l = kcurrLenMan[i] - 1;
                            kdistancesMan[i][kcurrLenMan[i]] =
                                    kdistancesMan[i][kcurrLenMan[i] - 1];
                            kneighborsMan[i][kcurrLenMan[i]] =
                                    kneighborsMan[i][kcurrLenMan[i] - 1];
                            while ((l >= 1) && distancesMan[i][j]
                                    < kdistancesMan[i][l - 1]) {
                                kdistancesMan[i][l] = kdistancesMan[i][l - 1];
                                kneighborsMan[i][l] = kneighborsMan[i][l - 1];
                                l--;
                            }
                            kdistancesMan[i][l] = distancesMan[i][j];
                            kneighborsMan[i][l] = i + j + 1;
                            kcurrLenMan[i]++;
                        } else {
                            kdistancesMan[i][kcurrLenMan[i]] =
                                    distancesMan[i][j];
                            kneighborsMan[i][kcurrLenMan[i]] = i + j + 1;
                            kcurrLenMan[i]++;
                        }
                    }
                } else {
                    kdistancesMan[i][0] = distancesMan[i][j];
                    kneighborsMan[i][0] = i + j + 1;
                    kcurrLenMan[i] = 1;
                }

                if (kcurrLenMan[other] > 0) {
                    if (kcurrLenMan[other] == maxK) {
                        if (distancesMan[i][j] < kdistancesMan[other][
                                kcurrLenMan[other] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesMan[i][j]
                                    < kdistancesMan[other][l - 1]) {
                                kdistancesMan[other][l] =
                                        kdistancesMan[other][l - 1];
                                kneighborsMan[other][l] =
                                        kneighborsMan[other][l - 1];
                                l--;
                            }
                            kdistancesMan[other][l] = distancesMan[i][j];
                            kneighborsMan[other][l] = i;
                        }
                    } else {
                        if (distancesMan[i][j]
                                < kdistancesMan[other][
                                kcurrLenMan[other] - 1]) {
                            // Search and insert.
                            l = kcurrLenMan[other] - 1;
                            kdistancesMan[other][kcurrLenMan[other]] =
                                    kdistancesMan[other][
                                    kcurrLenMan[other] - 1];
                            kneighborsMan[other][kcurrLenMan[other]] =
                                    kneighborsMan[other][
                                    kcurrLenMan[other] - 1];
                            while ((l >= 1) && distancesMan[i][j]
                                    < kdistancesMan[other][l - 1]) {
                                kdistancesMan[other][l] =
                                        kdistancesMan[other][l - 1];
                                kneighborsMan[other][l] =
                                        kneighborsMan[other][l - 1];
                                l--;
                            }
                            kdistancesMan[other][l] = distancesMan[i][j];
                            kneighborsMan[other][l] = i;
                            kcurrLenMan[other]++;
                        } else {
                            kdistancesMan[other][kcurrLenMan[other]] =
                                    distancesMan[i][j];
                            kneighborsMan[other][kcurrLenMan[other]] = i;
                            kcurrLenMan[other]++;
                        }
                    }
                } else {
                    kdistancesMan[other][0] = distancesMan[i][j];
                    kneighborsMan[other][0] = i;
                    kcurrLenMan[other] = 1;
                }
            }

            for (int i = 0; i < index; i++) {
                int j = index - i - 1;
                int other = index;
                if (kcurrLenEuc[i] > 0) {
                    if (kcurrLenEuc[i] == maxK) {
                        if (distancesEuc[i][j]
                                < kdistancesEuc[i][kcurrLenEuc[i] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesEuc[i][j]
                                    < kdistancesEuc[i][l - 1]) {
                                kdistancesEuc[i][l] = kdistancesEuc[i][l - 1];
                                kneighborsEuc[i][l] = kneighborsEuc[i][l - 1];
                                l--;
                            }
                            kdistancesEuc[i][l] = distancesEuc[i][j];
                            kneighborsEuc[i][l] = i + j + 1;
                        }
                    } else {
                        if (distancesEuc[i][j] < kdistancesEuc[i][
                                kcurrLenEuc[i] - 1]) {
                            // Search and insert.
                            l = kcurrLenEuc[i] - 1;
                            kdistancesEuc[i][kcurrLenEuc[i]] =
                                    kdistancesEuc[i][kcurrLenEuc[i] - 1];
                            kneighborsEuc[i][kcurrLenEuc[i]] =
                                    kneighborsEuc[i][kcurrLenEuc[i] - 1];
                            while ((l >= 1) && distancesEuc[i][j]
                                    < kdistancesEuc[i][l - 1]) {
                                kdistancesEuc[i][l] = kdistancesEuc[i][l - 1];
                                kneighborsEuc[i][l] = kneighborsEuc[i][l - 1];
                                l--;
                            }
                            kdistancesEuc[i][l] = distancesEuc[i][j];
                            kneighborsEuc[i][l] = i + j + 1;
                            kcurrLenEuc[i]++;
                        } else {
                            kdistancesEuc[i][kcurrLenEuc[i]] =
                                    distancesEuc[i][j];
                            kneighborsEuc[i][kcurrLenEuc[i]] = i + j + 1;
                            kcurrLenEuc[i]++;
                        }
                    }
                } else {
                    kdistancesEuc[i][0] = distancesEuc[i][j];
                    kneighborsEuc[i][0] = i + j + 1;
                    kcurrLenEuc[i] = 1;
                }

                if (kcurrLenEuc[other] > 0) {
                    if (kcurrLenEuc[other] == maxK) {
                        if (distancesEuc[i][j]
                                < kdistancesEuc[other][
                                kcurrLenEuc[other] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesEuc[i][j]
                                    < kdistancesEuc[other][l - 1]) {
                                kdistancesEuc[other][l] =
                                        kdistancesEuc[other][l - 1];
                                kneighborsEuc[other][l] =
                                        kneighborsEuc[other][l - 1];
                                l--;
                            }
                            kdistancesEuc[other][l] = distancesEuc[i][j];
                            kneighborsEuc[other][l] = i;
                        }
                    } else {
                        if (distancesEuc[i][j] < kdistancesEuc[other][
                                kcurrLenEuc[other] - 1]) {
                            // Search and insert.
                            l = kcurrLenEuc[other] - 1;
                            kdistancesEuc[other][kcurrLenEuc[other]] =
                                    kdistancesEuc[other][
                                    kcurrLenEuc[other] - 1];
                            kneighborsEuc[other][kcurrLenEuc[other]] =
                                    kneighborsEuc[other][
                                    kcurrLenEuc[other] - 1];
                            while ((l >= 1) && distancesEuc[i][j]
                                    < kdistancesEuc[other][l - 1]) {
                                kdistancesEuc[other][l] =
                                        kdistancesEuc[other][l - 1];
                                kneighborsEuc[other][l] =
                                        kneighborsEuc[other][l - 1];
                                l--;
                            }
                            kdistancesEuc[other][l] = distancesEuc[i][j];
                            kneighborsEuc[other][l] = i;
                            kcurrLenEuc[other]++;
                        } else {
                            kdistancesEuc[other][kcurrLenEuc[other]] =
                                    distancesEuc[i][j];
                            kneighborsEuc[other][kcurrLenEuc[other]] = i;
                            kcurrLenEuc[other]++;
                        }
                    }
                } else {
                    kdistancesEuc[other][0] = distancesEuc[i][j];
                    kneighborsEuc[other][0] = i;
                    kcurrLenEuc[other] = 1;
                }
            }

            for (int i = 0; i < index; i++) {
                int j = index - i - 1;
                int other = index;
                if (kcurrLenFrac[i] > 0) {
                    if (kcurrLenFrac[i] == maxK) {
                        if (distancesFrac[i][j]
                                < kdistancesFrac[i][kcurrLenFrac[i] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesFrac[i][j]
                                    < kdistancesFrac[i][l - 1]) {
                                kdistancesFrac[i][l] = kdistancesFrac[i][l - 1];
                                kneighborsFrac[i][l] = kneighborsFrac[i][l - 1];
                                l--;
                            }
                            kdistancesFrac[i][l] = distancesFrac[i][j];
                            kneighborsFrac[i][l] = i + j + 1;
                        }
                    } else {
                        if (distancesFrac[i][j]
                                < kdistancesFrac[i][kcurrLenFrac[i] - 1]) {
                            // Search and insert.
                            l = kcurrLenFrac[i] - 1;
                            kdistancesFrac[i][kcurrLenFrac[i]] =
                                    kdistancesFrac[i][kcurrLenFrac[i] - 1];
                            kneighborsFrac[i][kcurrLenFrac[i]] =
                                    kneighborsFrac[i][kcurrLenFrac[i] - 1];
                            while ((l >= 1) && distancesFrac[i][j]
                                    < kdistancesFrac[i][l - 1]) {
                                kdistancesFrac[i][l] = kdistancesFrac[i][l - 1];
                                kneighborsFrac[i][l] = kneighborsFrac[i][l - 1];
                                l--;
                            }
                            kdistancesFrac[i][l] = distancesFrac[i][j];
                            kneighborsFrac[i][l] = i + j + 1;
                            kcurrLenFrac[i]++;
                        } else {
                            kdistancesFrac[i][kcurrLenFrac[i]] =
                                    distancesFrac[i][j];
                            kneighborsFrac[i][kcurrLenFrac[i]] = i + j + 1;
                            kcurrLenFrac[i]++;
                        }
                    }
                } else {
                    kdistancesFrac[i][0] = distancesFrac[i][j];
                    kneighborsFrac[i][0] = i + j + 1;
                    kcurrLenFrac[i] = 1;
                }

                if (kcurrLenFrac[other] > 0) {
                    if (kcurrLenFrac[other] == maxK) {
                        if (distancesFrac[i][j]
                                < kdistancesFrac[other][
                                kcurrLenFrac[other] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesFrac[i][j]
                                    < kdistancesFrac[other][l - 1]) {
                                kdistancesFrac[other][l] =
                                        kdistancesFrac[other][l - 1];
                                kneighborsFrac[other][l] =
                                        kneighborsFrac[other][l - 1];
                                l--;
                            }
                            kdistancesFrac[other][l] = distancesFrac[i][j];
                            kneighborsFrac[other][l] = i;
                        }
                    } else {
                        if (distancesFrac[i][j]
                                < kdistancesFrac[other][
                                kcurrLenFrac[other] - 1]) {
                            // Search and insert.
                            l = kcurrLenFrac[other] - 1;
                            kdistancesFrac[other][kcurrLenFrac[other]] =
                                    kdistancesFrac[other][
                                    kcurrLenFrac[other] - 1];
                            kneighborsFrac[other][kcurrLenFrac[other]] =
                                    kneighborsFrac[other][
                                    kcurrLenFrac[other] - 1];
                            while ((l >= 1) && distancesFrac[i][j]
                                    < kdistancesFrac[other][l - 1]) {
                                kdistancesFrac[other][l] =
                                        kdistancesFrac[other][l - 1];
                                kneighborsFrac[other][l] =
                                        kneighborsFrac[other][l - 1];
                                l--;
                            }
                            kdistancesFrac[other][l] = distancesFrac[i][j];
                            kneighborsFrac[other][l] = i;
                            kcurrLenFrac[other]++;
                        } else {
                            kdistancesFrac[other][kcurrLenFrac[other]] =
                                    distancesFrac[i][j];
                            kneighborsFrac[other][kcurrLenFrac[other]] = i;
                            kcurrLenFrac[other]++;
                        }
                    }
                } else {
                    kdistancesFrac[other][0] = distancesFrac[i][j];
                    kneighborsFrac[other][0] = i;
                    kcurrLenFrac[other] = 1;
                }
            }

        }
        // Now we move to the next operating mode, where full stats are
        // calculated after each and every insertion.
        DataInstance currInstanceMan;
        DataInstance currInstanceEuc;
        DataInstance currInstanceFrac;
        for (int index = minInst; index < maxInst; index++) {
            // Get the current instances.
            currInstanceMan = resultsManh.data.get(index - minInst);
            currInstanceEuc = resultsEuc.data.get(index - minInst);
            currInstanceFrac = resultsFrac.data.get(index - minInst);
            // Get the current medoids.
            medoidsMan[index] = dset.getMedoidUpUntilIdex(centroid, cmetMan,
                    index + 1);
            medoidsEuc[index] = dset.getMedoidUpUntilIdex(centroid, cmetEuc,
                    index + 1);
            medoidsFrac[index] = dset.getMedoidUpUntilIdex(centroid, cmetFrac,
                    index + 1);
            // Calculate the relative contrast and variance for distance
            // concentration.
            ConcentrationCalculator ccMan =
                    new ConcentrationCalculator(dset, distancesMan);
            ConcentrationCalculator ccEuc =
                    new ConcentrationCalculator(dset, distancesEuc);
            ConcentrationCalculator ccFrac =
                    new ConcentrationCalculator(dset, distancesFrac);
            ccMan.calculateMeasures(index + 1);
            currInstanceMan.fAttr[0] = (float) ccMan.getRelativeContrast();
            currInstanceMan.fAttr[1] = (float) ccMan.getRelativeVariance();
            ccEuc.calculateMeasures(index + 1);
            currInstanceEuc.fAttr[0] = (float) ccEuc.getRelativeContrast();
            currInstanceEuc.fAttr[1] = (float) ccEuc.getRelativeVariance();
            ccFrac.calculateMeasures(index + 1);
            currInstanceFrac.fAttr[0] = (float) ccFrac.getRelativeContrast();
            currInstanceFrac.fAttr[1] = (float) ccFrac.getRelativeVariance();
            // Update the average distance.
            float avgDistMan = (float) ccMan.getMeanDist();
            float avgDistEuc = (float) ccEuc.getMeanDist();
            float avgDistFrac = (float) ccFrac.getMeanDist();
            // Update the k-nearest neighbor sets for all distance measures.
            int l;
            for (int i = 0; i < index; i++) {
                int j = index - i - 1;
                int other = index;
                if (kcurrLenMan[i] > 0) {
                    if (kcurrLenMan[i] == maxK) {
                        if (distancesMan[i][j]
                                < kdistancesMan[i][kcurrLenMan[i] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesMan[i][j]
                                    < kdistancesMan[i][l - 1]) {
                                kdistancesMan[i][l] = kdistancesMan[i][l - 1];
                                kneighborsMan[i][l] = kneighborsMan[i][l - 1];
                                l--;
                            }
                            kdistancesMan[i][l] = distancesMan[i][j];
                            kneighborsMan[i][l] = i + j + 1;
                        }
                    } else {
                        if (distancesMan[i][j] < kdistancesMan[i][
                                kcurrLenMan[i] - 1]) {
                            // Search and insert.
                            l = kcurrLenMan[i] - 1;
                            kdistancesMan[i][kcurrLenMan[i]] =
                                    kdistancesMan[i][kcurrLenMan[i] - 1];
                            kneighborsMan[i][kcurrLenMan[i]] =
                                    kneighborsMan[i][kcurrLenMan[i] - 1];
                            while ((l >= 1) && distancesMan[i][j]
                                    < kdistancesMan[i][l - 1]) {
                                kdistancesMan[i][l] = kdistancesMan[i][l - 1];
                                kneighborsMan[i][l] = kneighborsMan[i][l - 1];
                                l--;
                            }
                            kdistancesMan[i][l] = distancesMan[i][j];
                            kneighborsMan[i][l] = i + j + 1;
                            kcurrLenMan[i]++;
                        } else {
                            kdistancesMan[i][kcurrLenMan[i]] =
                                    distancesMan[i][j];
                            kneighborsMan[i][kcurrLenMan[i]] = i + j + 1;
                            kcurrLenMan[i]++;
                        }
                    }
                } else {
                    kdistancesMan[i][0] = distancesMan[i][j];
                    kneighborsMan[i][0] = i + j + 1;
                    kcurrLenMan[i] = 1;
                }

                if (kcurrLenMan[other] > 0) {
                    if (kcurrLenMan[other] == maxK) {
                        if (distancesMan[i][j]
                                < kdistancesMan[other][
                                kcurrLenMan[other] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesMan[i][j]
                                    < kdistancesMan[other][l - 1]) {
                                kdistancesMan[other][l] =
                                        kdistancesMan[other][l - 1];
                                kneighborsMan[other][l] =
                                        kneighborsMan[other][l - 1];
                                l--;
                            }
                            kdistancesMan[other][l] = distancesMan[i][j];
                            kneighborsMan[other][l] = i;
                        }
                    } else {
                        if (distancesMan[i][j]
                                < kdistancesMan[other][
                                kcurrLenMan[other] - 1]) {
                            // Search and insert.
                            l = kcurrLenMan[other] - 1;
                            kdistancesMan[other][kcurrLenMan[other]] =
                                    kdistancesMan[other][
                                    kcurrLenMan[other] - 1];
                            kneighborsMan[other][kcurrLenMan[other]] =
                                    kneighborsMan[other][
                                    kcurrLenMan[other] - 1];
                            while ((l >= 1) && distancesMan[i][j]
                                    < kdistancesMan[other][l - 1]) {
                                kdistancesMan[other][l] =
                                        kdistancesMan[other][l - 1];
                                kneighborsMan[other][l] =
                                        kneighborsMan[other][l - 1];
                                l--;
                            }
                            kdistancesMan[other][l] = distancesMan[i][j];
                            kneighborsMan[other][l] = i;
                            kcurrLenMan[other]++;
                        } else {
                            kdistancesMan[other][kcurrLenMan[other]] =
                                    distancesMan[i][j];
                            kneighborsMan[other][kcurrLenMan[other]] = i;
                            kcurrLenMan[other]++;
                        }
                    }
                } else {
                    kdistancesMan[other][0] = distancesMan[i][j];
                    kneighborsMan[other][0] = i;
                    kcurrLenMan[other] = 1;
                }
            }

            for (int i = 0; i < index; i++) {
                int j = index - i - 1;
                int other = index;
                if (kcurrLenEuc[i] > 0) {
                    if (kcurrLenEuc[i] == maxK) {
                        if (distancesEuc[i][j]
                                < kdistancesEuc[i][kcurrLenEuc[i] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesEuc[i][j]
                                    < kdistancesEuc[i][l - 1]) {
                                kdistancesEuc[i][l] = kdistancesEuc[i][l - 1];
                                kneighborsEuc[i][l] = kneighborsEuc[i][l - 1];
                                l--;
                            }
                            kdistancesEuc[i][l] = distancesEuc[i][j];
                            kneighborsEuc[i][l] = i + j + 1;
                        }
                    } else {
                        if (distancesEuc[i][j]
                                < kdistancesEuc[i][kcurrLenEuc[i] - 1]) {
                            // Search and insert.
                            l = kcurrLenEuc[i] - 1;
                            kdistancesEuc[i][kcurrLenEuc[i]] =
                                    kdistancesEuc[i][kcurrLenEuc[i] - 1];
                            kneighborsEuc[i][kcurrLenEuc[i]] =
                                    kneighborsEuc[i][kcurrLenEuc[i] - 1];
                            while ((l >= 1) && distancesEuc[i][j]
                                    < kdistancesEuc[i][l - 1]) {
                                kdistancesEuc[i][l] = kdistancesEuc[i][l - 1];
                                kneighborsEuc[i][l] = kneighborsEuc[i][l - 1];
                                l--;
                            }
                            kdistancesEuc[i][l] = distancesEuc[i][j];
                            kneighborsEuc[i][l] = i + j + 1;
                            kcurrLenEuc[i]++;
                        } else {
                            kdistancesEuc[i][kcurrLenEuc[i]] =
                                    distancesEuc[i][j];
                            kneighborsEuc[i][kcurrLenEuc[i]] = i + j + 1;
                            kcurrLenEuc[i]++;
                        }
                    }
                } else {
                    kdistancesEuc[i][0] = distancesEuc[i][j];
                    kneighborsEuc[i][0] = i + j + 1;
                    kcurrLenEuc[i] = 1;
                }

                if (kcurrLenEuc[other] > 0) {
                    if (kcurrLenEuc[other] == maxK) {
                        if (distancesEuc[i][j]
                                < kdistancesEuc[other][
                                kcurrLenEuc[other] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesEuc[i][j]
                                    < kdistancesEuc[other][l - 1]) {
                                kdistancesEuc[other][l] =
                                        kdistancesEuc[other][l - 1];
                                kneighborsEuc[other][l] =
                                        kneighborsEuc[other][l - 1];
                                l--;
                            }
                            kdistancesEuc[other][l] = distancesEuc[i][j];
                            kneighborsEuc[other][l] = i;
                        }
                    } else {
                        if (distancesEuc[i][j]
                                < kdistancesEuc[other][
                                kcurrLenEuc[other] - 1]) {
                            // Search and insert.
                            l = kcurrLenEuc[other] - 1;
                            kdistancesEuc[other][kcurrLenEuc[other]] =
                                    kdistancesEuc[other][
                                    kcurrLenEuc[other] - 1];
                            kneighborsEuc[other][kcurrLenEuc[other]] =
                                    kneighborsEuc[other][
                                    kcurrLenEuc[other] - 1];
                            while ((l >= 1) && distancesEuc[i][j]
                                    < kdistancesEuc[other][l - 1]) {
                                kdistancesEuc[other][l] =
                                        kdistancesEuc[other][l - 1];
                                kneighborsEuc[other][l] =
                                        kneighborsEuc[other][l - 1];
                                l--;
                            }
                            kdistancesEuc[other][l] = distancesEuc[i][j];
                            kneighborsEuc[other][l] = i;
                            kcurrLenEuc[other]++;
                        } else {
                            kdistancesEuc[other][kcurrLenEuc[other]] =
                                    distancesEuc[i][j];
                            kneighborsEuc[other][kcurrLenEuc[other]] = i;
                            kcurrLenEuc[other]++;
                        }
                    }
                } else {
                    kdistancesEuc[other][0] = distancesEuc[i][j];
                    kneighborsEuc[other][0] = i;
                    kcurrLenEuc[other] = 1;
                }
            }

            for (int i = 0; i < index; i++) {
                int j = index - i - 1;
                int other = index;
                if (kcurrLenFrac[i] > 0) {
                    if (kcurrLenFrac[i] == maxK) {
                        if (distancesFrac[i][j]
                                < kdistancesFrac[i][kcurrLenFrac[i] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesFrac[i][j]
                                    < kdistancesFrac[i][l - 1]) {
                                kdistancesFrac[i][l] = kdistancesFrac[i][l - 1];
                                kneighborsFrac[i][l] = kneighborsFrac[i][l - 1];
                                l--;
                            }
                            kdistancesFrac[i][l] = distancesFrac[i][j];
                            kneighborsFrac[i][l] = i + j + 1;
                        }
                    } else {
                        if (distancesFrac[i][j]
                                < kdistancesFrac[i][kcurrLenFrac[i] - 1]) {
                            // Search and insert.
                            l = kcurrLenFrac[i] - 1;
                            kdistancesFrac[i][kcurrLenFrac[i]] =
                                    kdistancesFrac[i][kcurrLenFrac[i] - 1];
                            kneighborsFrac[i][kcurrLenFrac[i]] =
                                    kneighborsFrac[i][kcurrLenFrac[i] - 1];
                            while ((l >= 1) && distancesFrac[i][j]
                                    < kdistancesFrac[i][l - 1]) {
                                kdistancesFrac[i][l] = kdistancesFrac[i][l - 1];
                                kneighborsFrac[i][l] = kneighborsFrac[i][l - 1];
                                l--;
                            }
                            kdistancesFrac[i][l] = distancesFrac[i][j];
                            kneighborsFrac[i][l] = i + j + 1;
                            kcurrLenFrac[i]++;
                        } else {
                            kdistancesFrac[i][kcurrLenFrac[i]] =
                                    distancesFrac[i][j];
                            kneighborsFrac[i][kcurrLenFrac[i]] = i + j + 1;
                            kcurrLenFrac[i]++;
                        }
                    }
                } else {
                    kdistancesFrac[i][0] = distancesFrac[i][j];
                    kneighborsFrac[i][0] = i + j + 1;
                    kcurrLenFrac[i] = 1;
                }

                if (kcurrLenFrac[other] > 0) {
                    if (kcurrLenFrac[other] == maxK) {
                        if (distancesFrac[i][j]
                                < kdistancesFrac[other][
                                kcurrLenFrac[other] - 1]) {
                            // Search and insert.
                            l = maxK - 1;
                            while ((l >= 1) && distancesFrac[i][j]
                                    < kdistancesFrac[other][l - 1]) {
                                kdistancesFrac[other][l] =
                                        kdistancesFrac[other][l - 1];
                                kneighborsFrac[other][l] =
                                        kneighborsFrac[other][l - 1];
                                l--;
                            }
                            kdistancesFrac[other][l] = distancesFrac[i][j];
                            kneighborsFrac[other][l] = i;
                        }
                    } else {
                        if (distancesFrac[i][j]
                                < kdistancesFrac[other][
                                kcurrLenFrac[other] - 1]) {
                            // Search and insert.
                            l = kcurrLenFrac[other] - 1;
                            kdistancesFrac[other][kcurrLenFrac[other]] =
                                    kdistancesFrac[other][
                                    kcurrLenFrac[other] - 1];
                            kneighborsFrac[other][kcurrLenFrac[other]] =
                                    kneighborsFrac[other][
                                    kcurrLenFrac[other] - 1];
                            while ((l >= 1) && distancesFrac[i][j]
                                    < kdistancesFrac[other][l - 1]) {
                                kdistancesFrac[other][l] =
                                        kdistancesFrac[other][l - 1];
                                kneighborsFrac[other][l] =
                                        kneighborsFrac[other][l - 1];
                                l--;
                            }
                            kdistancesFrac[other][l] = distancesFrac[i][j];
                            kneighborsFrac[other][l] = i;
                            kcurrLenFrac[other]++;
                        } else {
                            kdistancesFrac[other][kcurrLenFrac[other]] =
                                    distancesFrac[i][j];
                            kneighborsFrac[other][kcurrLenFrac[other]] = i;
                            kcurrLenFrac[other]++;
                        }
                    }
                } else {
                    kdistancesFrac[other][0] = distancesFrac[i][j];
                    kneighborsFrac[other][0] = i;
                    kcurrLenFrac[other] = 1;
                }
            }
            // Now calculate the neihbor occurrence frequencies for all the
            // monitored k-values.
            kneighborFrequenciesMan = new float[index + 1];
            kneighborFrequenciesEuc = new float[index + 1];
            kneighborFrequenciesFrac = new float[index + 1];
            for (int k = 1; k <= maxK; k++) {
                for (int i = 0; i <= index; i++) {
                    kneighborFrequenciesMan[kneighborsMan[i][k - 1]]++;
                    kneighborFrequenciesEuc[kneighborsEuc[i][k - 1]]++;
                    kneighborFrequenciesFrac[kneighborsFrac[i][k - 1]]++;
                }
                // Determine which points are the major hubs for each of the
                // tested distance measures.
                int hubManIndex = 0;
                int hubEucIndex = 0;
                int hubFracIndex = 0;
                float maxManHubness = 0;
                float maxEucHubness = 0;
                float maxFracHubness = 0;
                for (int i = 0; i <= index; i++) {
                    if (kneighborFrequenciesMan[i] > maxManHubness) {
                        hubManIndex = i;
                        maxManHubness = kneighborFrequenciesMan[i];
                    }
                    if (kneighborFrequenciesEuc[i] > maxEucHubness) {
                        hubEucIndex = i;
                        maxEucHubness = kneighborFrequenciesEuc[i];
                    }
                    if (kneighborFrequenciesFrac[i] > maxFracHubness) {
                        hubFracIndex = i;
                        maxFracHubness = kneighborFrequenciesFrac[i];
                    }
                }
                // Calculate all the monitored distance measures and
                // correlations.
                // First the Manhattan distance case.
                float mDistMan = cmetMan.dist(medoidsMan[index], centroid);
                float hDistMan = cmetMan.dist(dset.data.get(hubManIndex),
                        centroid);
                float hmDistMan = cmetMan.dist(dset.data.get(hubManIndex),
                        medoidsMan[index]);
                currInstanceMan.fAttr[2 + 10 * (k - 1)] = hDistMan / mDistMan;
                currInstanceMan.fAttr[3 + 10 * (k - 1)] = hmDistMan;
                currInstanceMan.fAttr[4 + 10 * (k - 1)] =
                        hmDistMan / avgDistMan;
                currInstanceMan.fAttr[5 + 10 * (k - 1)] = hDistMan;
                currInstanceMan.fAttr[6 + 10 * (k - 1)] =
                        PearsonCorrelation.correlation(kneighborFrequenciesMan,
                        normsMan);
                densitiesMan = NeighborSetFinder.
                        getDataDensitiesByNormalizedRadiusForElementsUntil(
                        dset, cmetMan, k, kneighborsMan, index + 1);
                currInstanceMan.fAttr[7 + 10 * (k - 1)] = PearsonCorrelation.
                        correlation(kneighborFrequenciesMan, densitiesMan);
                currInstanceMan.fAttr[8 + 10 * (k - 1)] = PearsonCorrelation.
                        correlation(densitiesMan, normsMan);
                currInstanceMan.fAttr[9 + 10 * (k - 1)] = DistanceCorrelation.
                        correlation(kneighborFrequenciesMan, normsMan);
                currInstanceMan.fAttr[10 + 10 * (k - 1)] = DistanceCorrelation.
                        correlation(kneighborFrequenciesMan, densitiesMan);
                currInstanceMan.fAttr[11 + 10 * (k - 1)] =
                        (float) DistanceCorrelation.correlation(densitiesMan,
                        normsMan);
                // The Euclidean case.
                float mDistEuc = cmetEuc.dist(medoidsEuc[index], centroid);
                float hDistEuc = cmetEuc.dist(dset.data.get(hubEucIndex),
                        centroid);
                float hmDistEuc = cmetEuc.dist(dset.data.get(hubEucIndex),
                        medoidsEuc[index]);
                currInstanceEuc.fAttr[2 + 10 * (k - 1)] = hDistEuc / mDistEuc;
                currInstanceEuc.fAttr[3 + 10 * (k - 1)] = hmDistEuc;
                currInstanceEuc.fAttr[4 + 10 * (k - 1)] =
                        hmDistEuc / avgDistEuc;
                currInstanceEuc.fAttr[5 + 10 * (k - 1)] = hDistEuc;
                currInstanceEuc.fAttr[6 + 10 * (k - 1)] =
                        PearsonCorrelation.correlation(kneighborFrequenciesEuc,
                        normsEuc);
                densitiesEuc = NeighborSetFinder.
                        getDataDensitiesByNormalizedRadiusForElementsUntil(
                        dset, cmetEuc, k, kneighborsEuc, index + 1);
                currInstanceEuc.fAttr[7 + 10 * (k - 1)] = PearsonCorrelation.
                        correlation(kneighborFrequenciesEuc, densitiesEuc);
                currInstanceEuc.fAttr[8 + 10 * (k - 1)] = PearsonCorrelation.
                        correlation(densitiesEuc, normsEuc);
                currInstanceEuc.fAttr[9 + 10 * (k - 1)] = DistanceCorrelation.
                        correlation(kneighborFrequenciesEuc, normsEuc);
                currInstanceEuc.fAttr[10 + 10 * (k - 1)] = DistanceCorrelation.
                        correlation(kneighborFrequenciesEuc, densitiesEuc);
                currInstanceEuc.fAttr[11 + 10 * (k - 1)] =
                        (float) DistanceCorrelation.correlation(
                        densitiesEuc, normsEuc);
                // Finally, the fractional distance case.
                float mDistFrac = cmetFrac.dist(medoidsFrac[index], centroid);
                float hDistFrac = cmetFrac.dist(dset.data.get(hubFracIndex),
                        centroid);
                float hmDistFrac = cmetFrac.dist(dset.data.get(hubFracIndex),
                        medoidsFrac[index]);
                currInstanceFrac.fAttr[2 + 10 * (k - 1)] =
                        hDistFrac / mDistFrac;
                currInstanceFrac.fAttr[3 + 10 * (k - 1)] = hmDistFrac;
                currInstanceFrac.fAttr[4 + 10 * (k - 1)] =
                        hmDistFrac / avgDistFrac;
                currInstanceFrac.fAttr[5 + 10 * (k - 1)] = hDistFrac;
                currInstanceFrac.fAttr[6 + 10 * (k - 1)] = PearsonCorrelation.
                        correlation(kneighborFrequenciesFrac, normsFrac);
                densitiesFrac = NeighborSetFinder.
                        getDataDensitiesByNormalizedRadiusForElementsUntil(
                        dset, cmetFrac, k, kneighborsFrac, index + 1);
                currInstanceFrac.fAttr[7 + 10 * (k - 1)] = PearsonCorrelation.
                        correlation(kneighborFrequenciesFrac, densitiesFrac);
                currInstanceFrac.fAttr[8 + 10 * (k - 1)] = PearsonCorrelation.
                        correlation(densitiesFrac, normsFrac);
                currInstanceFrac.fAttr[9 + 10 * (k - 1)] = DistanceCorrelation.
                        correlation(kneighborFrequenciesFrac, normsFrac);
                currInstanceFrac.fAttr[10 + 10 * (k - 1)] = DistanceCorrelation.
                        correlation(kneighborFrequenciesFrac, densitiesFrac);
                currInstanceFrac.fAttr[11 + 10 * (k - 1)] =
                        (float) DistanceCorrelation.correlation(
                        densitiesFrac, normsFrac);
            }
            System.out.println("Finished with the instance " + index);
        }
        // Persist the results.
        File outManFile = new File(outDir, "resultsMan_allK.arff");
        FileUtil.createFile(outManFile);
        System.out.println("Writing results file " + outManFile.getPath());
        pers.save(resultsManh, outManFile.getPath(), null);
        File outEucFile = new File(outDir, "resultsEuc_allK.arff");
        FileUtil.createFile(outEucFile);
        System.out.println("Writing results file " + outEucFile.getPath());
        pers.save(resultsEuc, outEucFile.getPath(), null);
        File outFracFile = new File(outDir, "resultsFrac_allK.arff");
        FileUtil.createFile(outFracFile);
        System.out.println("Writing results file " + outFracFile.getPath());
        pers.save(resultsFrac, outFracFile.getPath(), null);

    }

    /**
     * This method runs the script for hubness localization in intrinsically
     * high-dimensional Gaussian data.
     *
     * @param args Command line parameters, as specified.
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-outDir", "Path to the output directory.",
                CommandLineParser.STRING, true, false);
        clp.addParam("-maxK", "Maximal neighborhood size to consider.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-minInst", "Minimal number of instances.",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-maxInst", "Maximal number of instances..",
                CommandLineParser.INTEGER, true, false);
        clp.addParam("-dim", "Synthetic data dimensionality.",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File outDir = new File((String) clp.getParamValues("-outDir").get(0));
        int maxK = (Integer) clp.getParamValues("-maxK").get(0);
        int minInst = (Integer) clp.getParamValues("-minInst").get(0);
        int maxInst = (Integer) clp.getParamValues("-maxInst").get(0);
        int dim = (Integer) clp.getParamValues("-dim").get(0);
        // Run the experiment.
        GaussianHubnessLocalizer ghl = new GaussianHubnessLocalizer(
                outDir, dim, minInst, maxInst, maxK);
        ghl.runHubnessLocalizationExperiment();
    }
}
