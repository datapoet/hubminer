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
package distances.primary;

import data.representation.DataInstance;
import data.representation.util.DataMineConstants;
import java.io.Serializable;

public class DifferenceVarianceMetric extends DistanceMeasure
implements Serializable {
    
    private static final long serialVersionUID = 1L;

    /**
     * This class measures the variance of differences between two aligned time
     * series. Therefore, time series with different trends become distant and
     * those that are equal up to a factor of scale become very close. The
     * variance is calculated in place, without creating a temp array.
     *
     * @author Nenad Tomasev <nenad.tomasev at gmail.com>
     */
    @Override
    public float dist(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float sum = 0;
        int countValid = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            countValid++;
            sum += (arrFirst[i] - arrSecond[i]);
        }
        if (countValid == 0) {
            return 0;
        }
        float avgDifference = sum / countValid;
        sum = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sum += Math.pow((avgDifference - (arrFirst[i] - arrSecond[i])), 2);
        }
        return (sum / countValid);
    }

    @Override
    public float dist(int[] arrFirst, int[] arrSecond)
            throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float sum = 0;
        int countValid = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            countValid++;
            sum += (arrFirst[i] - arrSecond[i]);
        }
        if (countValid == 0) {
            return 0;
        }
        float avgDifference = sum / countValid;
        sum = 0;
        for (int i = 0; i < arrFirst.length; i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sum += Math.pow((avgDifference - (arrFirst[i] - arrSecond[i])), 2);
        }
        return (sum / countValid);
    }

    public float dist(DataInstance first, DataInstance second)
            throws MetricException {
        DistanceMeasure.assertInstances(first, second);
        float sumFloats = 0;
        float sumInts = 0;
        int countValidFloats = 0;
        int countValidInts = 0;
        for (int i = 0; i < first.getNumFAtt(); i++) {
            if (!DataMineConstants.isAcceptableFloat(first.fAttr[i])
                    || !DataMineConstants.isAcceptableFloat(second.fAttr[i])) {
                continue;
            }
            countValidFloats++;
            sumFloats += (first.fAttr[i] - second.fAttr[i]);
        }
        for (int i = 0; i < first.getNumIAtt(); i++) {
            if (!DataMineConstants.isAcceptableInt(first.iAttr[i])
                    || !DataMineConstants.isAcceptableInt(second.iAttr[i])) {
                continue;
            }
            countValidInts++;
            sumInts += (first.iAttr[i] - second.iAttr[i]);
        }
        int countValid = countValidFloats + countValidInts;
        if (countValid == 0) {
            return 0;
        }
        float avgDifferenceFloats = (countValidFloats > 0)
                ? sumFloats / countValidFloats : 0;
        float avgDifferenceInts = (countValidInts > 0)
                ? sumInts / countValidInts : 0;
        float sum = 0;
        for (int i = 0; i < first.getNumFAtt(); i++) {
            if (!DataMineConstants.isAcceptableFloat(first.fAttr[i])
                    || !DataMineConstants.isAcceptableFloat(second.fAttr[i])) {
                continue;
            }
            countValidFloats++;
            sum += Math.pow((avgDifferenceFloats - (first.fAttr[i]
                    - second.fAttr[i])), 2);
        }
        for (int i = 0; i < first.getNumIAtt(); i++) {
            if (!DataMineConstants.isAcceptableInt(first.iAttr[i])
                    || !DataMineConstants.isAcceptableInt(second.iAttr[i])) {
                continue;
            }
            countValidInts++;
            sum += Math.pow((avgDifferenceInts - (first.iAttr[i]
                    - second.iAttr[i])), 2);
        }
        return (sum / countValid);
    }
}