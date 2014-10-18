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
package statistics;

import data.representation.DataInstance;
import data.representation.DataSet;

/**
 * Calculator for feature covariances.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class CovarianceFinder {

    private DataSet dset;
    // Covariance measures for the floats.
    private float[] featureMeansFloat;
    private float[][] deviationsFloat;
    private float[][] floatCovariance;
    // Covariance measures for the ints.
    private float[] featureMeansInt;
    private float[][] deviationsInt;
    private float[][] intCovariance;
    // IMPORTANT: the attributes are concatenated, first the float atts and then
    // the int atts. This holds for feature means, deviations and the covariance
    // matrix itself.
    private float[] featureMeansFI;
    private float[][] deviationsFI;
    private float[][] fiCovariance;

    /**
     * @param dset DataSet object to find the covariance of.
     */
    public CovarianceFinder(DataSet dset) {
        this.dset = dset;
    }

    /**
     * @return Float array of mean values of float features.
     */
    public float[] getFloatMeans() {
        return featureMeansFloat;
    }

    /**
     * @return Float array of mean values of integer features.
     */
    public float[] getIntMeans() {
        return featureMeansInt;
    }

    /**
     * Finds the covariance matrix for float and integer attributes.
     *
     * @return A 2d float array where the covariance is concatenated from the
     * floats and the ints. First floatSize columns/rows come from floats, then
     * intSize from ints follow.
     */
    public float[][] calculateFICovariance() {
        if (dset == null || dset.isEmpty()) {
            return null;
        }
        calculateFIEmpiricalMean();
        calculateFIDeviations();
        fiCovariance = new float[featureMeansFI.length][featureMeansFI.length];
        for (int i = 0; i < featureMeansFI.length; i++) {
            for (int j = i + 1; j < featureMeansFI.length; j++) {
                for (int k = 0; k < dset.size(); k++) {
                    fiCovariance[i][j] +=
                            deviationsFI[k][i] * deviationsFI[k][j];
                }
                fiCovariance[i][j] /= ((float) dset.size());
                fiCovariance[j][i] = fiCovariance[i][j];
            }
        }
        return fiCovariance;
    }

    /**
     * Finds the covariance matrix for float attributes.
     *
     * @return A 2d float array of float feature covariance.
     */
    public float[][] calculateFloatCovariance() {
        if (dset == null || dset.isEmpty()) {
            return null;
        }
        calculateFEmpiricalMean();
        calculateFDeviations();
        floatCovariance =
                new float[featureMeansFloat.length][featureMeansFloat.length];
        for (int i = 0; i < featureMeansFloat.length; i++) {
            for (int j = i + 1; j < featureMeansFloat.length; j++) {
                for (int k = 0; k < dset.size(); k++) {
                    floatCovariance[i][j] +=
                            deviationsFloat[k][i] * deviationsFloat[k][j];
                }
                floatCovariance[i][j] /= ((float) dset.size());
                floatCovariance[j][i] = floatCovariance[i][j];
            }
        }
        return floatCovariance;
    }

    /**
     * Finds the covariance matrix for integer attributes.
     *
     * @return A 2d float array of integer feature covariance.
     */
    public float[][] calculateIntCovariance() {
        if (dset == null || dset.isEmpty()) {
            return null;
        }
        calculateIEmpiricalMean();
        calculateIDeviations();
        intCovariance =
                new float[featureMeansInt.length][featureMeansInt.length];
        for (int i = 0; i < featureMeansInt.length; i++) {
            for (int j = i + 1; j < featureMeansInt.length; j++) {
                for (int k = 0; k < dset.size(); k++) {
                    intCovariance[i][j] +=
                            deviationsInt[k][i] * deviationsInt[k][j];
                }
                intCovariance[i][j] /= ((float) dset.size());
                intCovariance[j][i] = intCovariance[i][j];
            }
        }
        return intCovariance;
    }

    /**
     * Calculate absolute deviations for float members.
     */
    private void calculateFDeviations() {
        if (!dset.hasFloatAttr()) {
            return;
        }
        DataInstance instance;
        deviationsFloat = new float[dset.size()][featureMeansFloat.length];
        for (int i = 0; i < dset.size(); i++) {
            instance = dset.data.get(i);
            for (int j = 0; j < featureMeansFloat.length; j++) {
                deviationsFloat[i][j] =
                        instance.fAttr[j] - featureMeansFloat[j];
            }
        }
    }

    /**
     * Calculates the mean for each float dimension.
     */
    private void calculateFEmpiricalMean() {
        if (dset.hasFloatAttr()) {
            featureMeansFloat = new float[dset.getNumFloatAttr()];
        } else {
            return;
        }
        DataInstance instance;
        for (int i = 0; i < dset.size(); i++) {
            instance = dset.data.get(i);
            for (int j = 0; j < dset.fAttrNames.length; j++) {
                featureMeansFloat[j] += instance.fAttr[j];
            }
        }
        for (int j = 0; j < dset.fAttrNames.length; j++) {
            featureMeansFloat[j] /= (float) dset.size();
        }
    }

    /**
     * Calculate absolute deviations for integer members.
     */
    private void calculateIDeviations() {
        if (!dset.hasIntAttr()) {
            return;
        }
        DataInstance instance;
        deviationsInt = new float[dset.size()][featureMeansInt.length];
        for (int i = 0; i < dset.size(); i++) {
            instance = dset.data.get(i);
            for (int j = 0; j < featureMeansInt.length; j++) {
                deviationsInt[i][j] = instance.iAttr[j] - featureMeansInt[j];
            }
        }
    }

    /**
     * Calculates the mean for each integer dimension.
     */
    private void calculateIEmpiricalMean() {
        if (dset.hasIntAttr()) {
            featureMeansInt = new float[dset.getNumIntAttr()];
        } else {
            return;
        }
        DataInstance instance;
        for (int i = 0; i < dset.size(); i++) {
            instance = dset.data.get(i);
            for (int j = 0; j < dset.iAttrNames.length; j++) {
                featureMeansInt[j] += (float) (instance.iAttr[j]);
            }
        }
        for (int j = 0; j < dset.iAttrNames.length; j++) {
            featureMeansInt[j] /= (float) dset.size();
        }
    }

    /**
     * Calculate absolute deviations for float and integer members.
     */
    private void calculateFIDeviations() {
        int intSize = dset.getNumIntAttr();
        int floatSize = dset.getNumFloatAttr();
        DataInstance instance;
        deviationsInt = new float[dset.size()][featureMeansFI.length];
        for (int i = 0; i < dset.size(); i++) {
            instance = dset.data.get(i);
            for (int j = 0; j < floatSize; j++) {
                deviationsInt[i][j] = instance.fAttr[j] - featureMeansFI[j];
            }
            for (int j = 0; j < intSize; j++) {
                deviationsInt[i][floatSize + j] =
                        instance.iAttr[j] - featureMeansFI[j];
            }
        }
    }

    /**
     * Calculates the mean for all float and integer features.
     */
    private void calculateFIEmpiricalMean() {
        int intSize = dset.getNumIntAttr();
        int floatSize = dset.getNumFloatAttr();
        featureMeansFI = new float[floatSize + intSize];
        DataInstance instance;
        for (int i = 0; i < dset.size(); i++) {
            instance = dset.data.get(i);
            for (int j = 0; j < floatSize; j++) {
                featureMeansFI[j] += instance.fAttr[j];
            }
            for (int j = 0; j < intSize; j++) {
                featureMeansFI[floatSize + j] += (float) (instance.iAttr[j]);
            }
        }
        for (int j = 0; j < featureMeansFI.length; j++) {
            featureMeansFI[j] /= (float) dset.size();
        }
    }
}
