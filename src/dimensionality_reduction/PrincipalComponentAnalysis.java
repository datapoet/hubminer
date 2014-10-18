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
package dimensionality_reduction;

import data.representation.DataInstance;
import data.representation.DataSet;
import data.representation.util.DataMineConstants;
import ioformat.IOARFF;
import ioformat.SupervisedLoader;
import transformation.TransformationInterface;
import util.CommandLineParser;

import java.io.File;
import java.util.Random;

/**
 * Implements the PCA algorithm for dimensionality reduction.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class PrincipalComponentAnalysis implements TransformationInterface {

    // Number of principal components.
    private int numPrincipalComponents;
    // Original dataset to reduce the dimensionality of.
    private DataSet ds;
    // Transformed DataSet of reduced dimensionality.
    private DataSet transformedData;
    // Differences between feature values in data instances and the average
    // feature values.
    private double[][] diffs;
    // Average feature values.
    private double[] featureAverages;
    // Principal components that span the space the data should be projected on.
    private double[][] principalComponents;
    // Covariance matrix of the original data.
    private double[][] covMat;
    // Current temporary eigenvector.
    private double[] eigenTemp;
    // Error of the current iteration used to check for convergence progress.
    private double error;

    /**
     * @constructor @param ds DataSet - the original data set.
     * @param numPrincipalComponents Number of desired principal components.
     */
    public PrincipalComponentAnalysis(DataSet ds, int numPrincipalComponents) {
        this.numPrincipalComponents = numPrincipalComponents;
        this.ds = ds;
    }

    /**
     * Set the original data to reduce dimensionality of.
     *
     * @param ds DataSet - the original data set.
     */
    @Override
    public void setData(DataSet ds) {
        this.ds = ds;
    }

    /**
     * @return String containing a link to the paper and the algorithm details.
     */
    public static String citing() {
        return "Transformation based on the fixed-point algorithm described "
                + "here: http://maxwell.me.gu.edu.au/spl/publications/papers/"
                + "prl07_alok_pca.pdf";
    }

    /**
     * @return DataSet prior to transformation.
     */
    public DataSet getOriginalData() {
        return ds;
    }

    /**
     * @return Covariance matrix of the original data calculated during the
     * initial phase of PCA.
     */
    public double[][] getCovarianceMatrix() {
        return covMat;
    }

    @Override
    public DataSet transformData() {
        findInitialStats();
        findPrincipalComponents();
        projectData();
        return transformedData;
    }

    /**
     * Projects the data onto the calculated lower-dimensional subspace. The
     * results are kept within the transformedData internal DataSet variable.
     */
    private void projectData() {
        transformedData = new DataSet();
        String[] fNames = new String[numPrincipalComponents];
        for (int i = 0; i < numPrincipalComponents; i++) {
            fNames[i] = "Principal Component " + i;
        }
        transformedData.fAttrNames = fNames;
        for (int i = 0; i < ds.size(); i++) {
            DataInstance instance = new DataInstance(transformedData);
            instance.embedInDataset(transformedData);
            transformedData.addDataInstance(instance);
            instance.setCategory(ds.getLabelOf(i));
        }
        double total;
        for (int i = 0; i < ds.size(); i++) {
            DataInstance instance = transformedData.getInstance(i);
            for (int d = 0; d < numPrincipalComponents; d++) {
                total = 0;
                for (int dorig = 0; dorig < ds.getNumFloatAttr(); dorig++) {
                    total += principalComponents[d][dorig] * diffs[i][dorig];
                }
                instance.fAttr[d] = (float) total;
            }
        }
    }

    /**
     * Calculate the error of the current iteration.
     *
     * @param d Index of the eigenvector that is currently being calculated.
     */
    private void updateError(int d) {
        double total = 0;
        for (int index = 0; index < eigenTemp.length; index++) {
            total += eigenTemp[index] * principalComponents[d][index];
        }
        error = Math.abs(Math.abs(total) - 1);
    }

    /**
     * Calculate feature averages, deviations of all feature vectors from the
     * averages and the covariance matrix.
     */
    private void findInitialStats() {
        int numFloats = ds.getNumFloatAttr();
        featureAverages = new double[numFloats];
        double average;
        for (int d = 0; d < ds.getNumFloatAttr(); d++) {
            // Average feature value in the given dimension.
            average = 0;
            for (int j = 0; j < ds.size(); j++) {
                average += ds.getInstance(j).fAttr[d];
            }
            average = average / ds.size();
            featureAverages[d] = average;
        }
        diffs = new double[ds.size()][numFloats];
        for (int i = 0; i < ds.size(); i++) {
            for (int j = 0; j < featureAverages.length; j++) {
                diffs[i][j] = ds.getInstance(i).fAttr[j] - featureAverages[j];
            }
        }
        covMat = new double[numFloats][numFloats];
        for (int i = 0; i < covMat.length; i++) {
            for (int j = 0; j < covMat[i].length; j++) {
                double total = 0;
                for (int index = 0; index < ds.size(); index++) {
                    total += diffs[index][i] * diffs[index][j];
                }
                total = total / ds.size();
                covMat[i][j] = total;
            }
        }
    }

    /**
     * Calculate the main eigenvectors that are the principal components of the
     * data.
     */
    private void findPrincipalComponents() {
        Random randa = new Random();
        int numFloats = ds.getNumFloatAttr();
        double total;
        principalComponents = new double[numPrincipalComponents][numFloats];
        for (int i = 0; i < principalComponents.length; i++) {
            for (int j = 0; j < principalComponents[i].length; j++) {
                principalComponents[i][j] = randa.nextDouble();
            }
        }
        eigenTemp = new double[numFloats];
        double[] vectTotal = new double[numFloats];
        for (int d = 0; d < numPrincipalComponents; d++) {
            error = 1;
            while (error > DataMineConstants.EPSILON_CONVERGENCE) {
                for (int i = 0; i < eigenTemp.length; i++) {
                    total = 0;
                    for (int j = 0; j < principalComponents[d].length; j++) {
                        total += principalComponents[d][j] * covMat[i][j];
                    }
                    eigenTemp[i] = total;
                }
                // Gram-Schmidt orthogonalization.
                for (int i = 0; i < vectTotal.length; i++) {
                    vectTotal[i] = 0;
                }
                for (int i = 0; i < d; i++) {
                    total = 0;
                    for (int j = 0; j < numFloats; j++) {
                        total += eigenTemp[j] * principalComponents[i][j];
                    }
                    for (int j = 0; j < numFloats; j++) {
                        vectTotal[j] += total * principalComponents[i][j];
                    }
                }
                for (int i = 0; i < numFloats; i++) {
                    eigenTemp[i] -= vectTotal[i];
                }
                total = 0;
                // Normalization.
                for (int i = 0; i < numFloats; i++) {
                    total += Math.pow(eigenTemp[i], 2);
                }
                total = Math.sqrt(total);
                for (int i = 0; i < eigenTemp.length; i++) {
                    eigenTemp[i] /= total;
                }
                updateError(d);
                for (int i = 0; i < numFloats; i++) {
                    principalComponents[d][i] = eigenTemp[i];
                }
            }
        }
    }

    /**
     * Performs PCA from the file specified by the user, reducing it to a
     * specified number of dimensions and persisting the results to an output
     * file.
     *
     * @param args
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        CommandLineParser clp = new CommandLineParser(true);
        clp.addParam("-inFile", "Path to the input dataset",
                CommandLineParser.STRING, true, false);
        clp.addParam("-outFile", "Output path", CommandLineParser.STRING,
                true, false);
        clp.addParam("-dim", "Dimensionality of data projection",
                CommandLineParser.INTEGER, true, false);
        clp.parseLine(args);
        File inFile = new File((String) (clp.getParamValues("-inFile").get(0)));
        File outFile = new File((String) (clp.getParamValues(
                "-outFile").get(0)));
        int numVectors = (Integer) (clp.getParamValues("-dim").get(0));
        DataSet inputSet = SupervisedLoader.loadData(inFile.getPath(), false);
        PrincipalComponentAnalysis pca = new PrincipalComponentAnalysis(
                inputSet, numVectors);
        DataSet output = pca.transformData();
        IOARFF saver = new IOARFF();
        saver.saveLabeled(output, outFile.getPath());
    }
}
