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
import data.representation.util.DataMineConstants;
import java.util.ArrayList;

/**
 * Calculates the moments of the distribution.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class HigherMoments {

    DataSet dset;

    /**
     * @param dset DataSet that will be analyzed.
     */
    public HigherMoments(DataSet dset) {
        this.dset = dset;
    }

    /**
     * @param fArray Float array.
     * @return Mean value of the array.
     */
    public static float calculateArrayMean(float[] fArray) {
        double sum = 0;
        for (int i = 0; i < fArray.length; i++) {
            sum += fArray[i];
        }
        sum /= (double) fArray.length;
        return (float) (sum);
    }

    /**
     * @param mean Mean value of the float array.
     * @param fArray Float array.
     * @return Standard deviation of the float array.
     */
    public static float calculateArrayStDev(float mean, float[] fArray) {
        double sum = 0;
        for (int i = 0; i < fArray.length; i++) {
            sum += Math.pow(Math.abs(mean - fArray[i]), 2);
        }
        sum /= (double) fArray.length;
        sum = Math.sqrt(sum);
        return (float) (sum);
    }

    /**
     * @param iArray Integer array.
     * @return Mean value of the integer array.
     */
    public static float calculateArrayMean(int[] iArray) {
        double sum = 0;
        for (int i = 0; i < iArray.length; i++) {
            sum += iArray[i];
        }
        sum /= (double) iArray.length;
        return (float) (sum);
    }

    /**
     * @param mean Mean value of the integer array.
     * @param iArray Integer array.
     * @return Standard deviation of the integer array.
     */
    public static float calculateArrayStDev(float mean, int[] iArray) {
        double sum = 0;
        for (int i = 0; i < iArray.length; i++) {
            sum += Math.pow(Math.abs(mean - iArray[i]), 2);
        }
        sum /= (double) iArray.length;
        sum = Math.sqrt(sum);
        return (float) (sum);
    }

    /**
     * Calculates the third standard moment of the provided feature
     * distribution.
     *
     * @param attType Integer value denoting attribute type.
     * @param attIndex Index of the attribute in the DataSet definition for that
     * type.
     * @return Skewness of the specified attribute.
     */
    public float calculateSkewForAttribute(int attType, int attIndex) {
        DataInstance instance;
        if (attType == DataMineConstants.FLOAT) {
            float mean = 0f;
            for (int i = 0; i < dset.size(); i++) {
                instance = dset.data.get(i);
                mean += instance.fAttr[attIndex];
            }
            mean /= (float) dset.size();
            float m3 = 0;
            float m2 = 0;
            float tmpSquare;
            float tmpVal;
            for (int i = 0; i < dset.size(); i++) {
                instance = dset.data.get(i);
                tmpVal = instance.fAttr[attIndex] - mean;
                tmpSquare = tmpVal * tmpVal;
                m2 += tmpSquare;
                m3 += tmpSquare * tmpVal;
            }
            m2 /= (float) dset.size();
            m3 /= (float) dset.size();
            float skew = m3 / (float) Math.pow(m2, 1.5f);
            return skew;
        } else if (attType == DataMineConstants.INTEGER) {
            float mean = 0f;
            for (int i = 0; i < dset.size(); i++) {
                instance = dset.data.get(i);
                mean += instance.iAttr[attIndex];
            }
            mean /= (float) dset.size();
            float m3 = 0;
            float m2 = 0;
            float tmpSquare;
            float tmpVal;
            for (int i = 0; i < dset.size(); i++) {
                instance = dset.data.get(i);
                tmpVal = instance.iAttr[attIndex] - mean;
                tmpSquare = tmpVal * tmpVal;
                m2 += tmpSquare;
                m3 += tmpSquare * tmpVal;
            }
            m2 /= (float) dset.size();
            m3 /= (float) dset.size();
            float skew = m3 / (float) Math.pow(m2, 1.5f);
            return skew;
        } else {
            return 0f;
        }

    }

    /**
     * Calculates the third standard moment of the provided value array.
     *
     * @param arr Float value array.
     * @return Skewness of the specified distribution.
     */
    public static float calculateSkewForSampleArray(float[] arr) {
        if (arr == null || arr.length == 0) {
            return 0f;
        }
        float mean = 0f;
        for (int i = 0; i < arr.length; i++) {
            mean += arr[i];
        }
        mean /= (float) arr.length;
        float m3 = 0;
        float m2 = 0;
        float tmpSquare;
        float tmpVal;
        for (int i = 0; i < arr.length; i++) {
            tmpVal = arr[i] - mean;
            tmpSquare = tmpVal * tmpVal;
            m2 += tmpSquare;
            m3 += tmpSquare * tmpVal;
        }
        m2 /= (float) arr.length;
        m3 /= (float) arr.length;
        float skew = m3 / (float) Math.pow(m2, 1.5f);
        return skew;
    }

    /**
     * Calculates the third standard moment of the provided value array.
     *
     * @param arr Integer value array.
     * @return Skewness of the specified distribution.
     */
    public static float calculateSkewForSampleArray(int[] arr) {
        if (arr == null || arr.length == 0) {
            return 0f;
        }
        float mean = 0f;
        for (int i = 0; i < arr.length; i++) {
            mean += arr[i];
        }
        mean /= (float) arr.length;
        float m3 = 0;
        float m2 = 0;
        float tmpSquare;
        float tmpVal;
        for (int i = 0; i < arr.length; i++) {
            tmpVal = arr[i] - mean;
            tmpSquare = tmpVal * tmpVal;
            m2 += tmpSquare;
            m3 += tmpSquare * tmpVal;
        }
        m2 /= (float) arr.length;
        m3 /= (float) arr.length;
        float skew = m3 / (float) Math.pow(m2, 1.5f);
        return skew;
    }

    /**
     * Calculates the fourth standard moment of the provided feature
     * distribution. It represents the steepness of the curve.
     *
     * @param attType Integer value denoting attribute type.
     * @param attIndex Index of the attribute in the DataSet definition for that
     * type.
     * @return Kurtosis of the specified attribute.
     */
    public float calculateKurtosisForAttribute(int attType, int attIndex) {
        DataInstance instance;
        if (attType == DataMineConstants.FLOAT) {
            float mean = 0f;
            for (int i = 0; i < dset.size(); i++) {
                instance = dset.data.get(i);
                mean += instance.fAttr[attIndex];
            }
            mean /= (float) dset.size();
            float m4 = 0;
            float m2 = 0;
            float tmpSquare;
            float tmpVal;
            for (int i = 0; i < dset.size(); i++) {
                instance = dset.data.get(i);
                tmpVal = instance.fAttr[attIndex] - mean;
                tmpSquare = tmpVal * tmpVal;
                m2 += tmpSquare;
                m4 += tmpSquare * tmpSquare;
            }
            m2 /= (float) dset.size();
            m4 /= (float) dset.size();
            float kurtosis = (m4 / (float) Math.pow(m2, 2)) - 3;
            return kurtosis;
        } else if (attType == DataMineConstants.INTEGER) {
            float mean = 0f;
            for (int i = 0; i < dset.size(); i++) {
                instance = dset.data.get(i);
                mean += instance.iAttr[attIndex];
            }
            mean /= (float) dset.size();
            float m4 = 0;
            float m2 = 0;
            float tmpSquare;
            float tmpVal;
            for (int i = 0; i < dset.size(); i++) {
                instance = dset.data.get(i);
                tmpVal = instance.iAttr[attIndex] - mean;
                tmpSquare = tmpVal * tmpVal;
                m2 += tmpSquare;
                m4 += tmpSquare * tmpSquare;
            }
            m2 /= (float) dset.size();
            m4 /= (float) dset.size();
            float kurtosis = (m4 / (float) Math.pow(m2, 2)) - 3;
            return kurtosis;
        } else {
            return 0f;
        }

    }

    /**
     * Calculates the fourth standard moment of the provided value array.
     *
     * @param arr Float value array.
     * @return Kurtosis of the specified distribution.
     */
    public static float calculateKurtosisForSampleArray(float[] arr) {
        if (arr == null || arr.length == 0) {
            return 0f;
        }
        float mean = 0f;
        for (int i = 0; i < arr.length; i++) {
            mean += arr[i];
        }
        mean /= (float) arr.length;
        float m4 = 0;
        float m2 = 0;
        float tmpSquare;
        float tmpVal;
        for (int i = 0; i < arr.length; i++) {
            tmpVal = arr[i] - mean;
            tmpSquare = tmpVal * tmpVal;
            m2 += tmpSquare;
            m4 += tmpSquare * tmpSquare;
        }
        m2 /= (float) arr.length;
        m4 /= (float) arr.length;
        float kurtosis = (m4 / (float) Math.pow(m2, 2)) - 3;
        return kurtosis;
    }

    /**
     * Calculates the fourth standard moment of the provided value array.
     *
     * @param arr Integer value array.
     * @return Kurtosis of the specified distribution.
     */
    public static float calculateKurtosisForSampleArray(int[] arr) {
        if (arr == null || arr.length == 0) {
            return 0f;
        }
        float mean = 0f;
        for (int i = 0; i < arr.length; i++) {
            mean += arr[i];
        }
        mean /= (float) arr.length;
        float m4 = 0;
        float m2 = 0;
        float tmpSquare;
        float tmpVal;
        for (int i = 0; i < arr.length; i++) {
            tmpVal = arr[i] - mean;
            tmpSquare = tmpVal * tmpVal;
            m2 += tmpSquare;
            m4 += tmpSquare * tmpSquare;
        }
        m2 /= (float) arr.length;
        m4 /= (float) arr.length;
        float kurtosis = (m4 / (float) Math.pow(m2, 2)) - 3;
        return kurtosis;
    }
    
    /**
     * @param fArray Float ArrayList.
     * @return Mean value of the ArrayList.
     */
    public static float calculateArrayListMean(ArrayList<Float> fArray) {
        double sum = 0;
        for (int i = 0; i < fArray.size(); i++) {
            sum += fArray.get(i);
        }
        sum /= (double) fArray.size();
        return (float) (sum);
    }

    /**
     * @param mean Mean value of the float ArrayList.
     * @param fArray Float ArrayList.
     * @return Standard deviation of the float ArrayList.
     */
    public static float calculateArrayListStDev(float mean,
            ArrayList<Float> fArray) {
        double sum = 0;
        for (int i = 0; i < fArray.size(); i++) {
            sum += Math.pow(Math.abs(mean - fArray.get(i)), 2);
        }
        sum /= (double) fArray.size();
        sum = Math.sqrt(sum);
        return (float) (sum);
    }
    
    /**
     * Calculates the third standard moment of the provided value ArrayList.
     *
     * @param arr Float value ArrayList.
     * @return Skewness of the specified distribution.
     */
    public static float calculateSkewForSampleArrayList(
            ArrayList<Float> fArray) {
        if (fArray == null || fArray.isEmpty()) {
            return 0f;
        }
        float mean = 0f;
        for (int i = 0; i < fArray.size(); i++) {
            mean += fArray.get(i);
        }
        mean /= (float) fArray.size();
        float m3 = 0;
        float m2 = 0;
        float tmpSquare;
        float tmpVal;
        for (int i = 0; i < fArray.size(); i++) {
            tmpVal = fArray.get(i) - mean;
            tmpSquare = tmpVal * tmpVal;
            m2 += tmpSquare;
            m3 += tmpSquare * tmpVal;
        }
        m2 /= (float) fArray.size();
        m3 /= (float) fArray.size();
        float skew = m3 / (float) Math.pow(m2, 1.5f);
        return skew;
    }
    
    /**
     * Calculates the fourth standard moment of the provided value ArrayList.
     *
     * @param arr Float value ArrayList.
     * @return Kurtosis of the specified distribution.
     */
    public static float calculateKurtosisForSampleArrayList(
            ArrayList<Float> fArray) {
        if (fArray == null || fArray.isEmpty()) {
            return 0f;
        }
        float mean = 0f;
        for (int i = 0; i < fArray.size(); i++) {
            mean += fArray.get(i);
        }
        mean /= (float) fArray.size();
        float m4 = 0;
        float m2 = 0;
        float tmpSquare;
        float tmpVal;
        for (int i = 0; i < fArray.size(); i++) {
            tmpVal = fArray.get(i) - mean;
            tmpSquare = tmpVal * tmpVal;
            m2 += tmpSquare;
            m4 += tmpSquare * tmpSquare;
        }
        m2 /= (float) fArray.size();
        m4 /= (float) fArray.size();
        float kurtosis = (m4 / (float) Math.pow(m2, 2)) - 3;
        return kurtosis;
    }
}
