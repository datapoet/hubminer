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

/**
 * This class implements a special case of the Minkowski metric on 3D data.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Minkowski3D extends DistanceMeasure implements Serializable {
    
    private static final long serialVersionUID = 1L;

    /**
     * Euclidean distances in 3D.
     *
     * @author Nenad Tomasev <nenad.tomasev at gmail.com>
     */
    private int p = 2;

    /**
     * @param p Defines the L_p Minkowski distance.
     */
    public Minkowski3D(int p) {
        this.p = p;
    }

    /**
     * Default is Euclidean
     */
    public Minkowski3D() {
        this.p = 2;
    }

    @Override
    public float dist(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float sum = 0;
        for (int i = 0; i < Math.min(3, arrFirst.length); i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sum += Math.pow(Math.abs(arrFirst[i] - arrSecond[i]), p);
        }
        sum = (float) Math.pow(sum, 1. / p);
        return sum;
    }

    /**
     * @param Float array.
     * @return Norm of the given array.
     */
    public float norm(float[] fVector) throws Exception {
        if (fVector == null) {
            return 0f;
        }
        float sum = 0f;
        for (int i = 0; i < Math.min(3, fVector.length); i++) {
            if (!DataMineConstants.isAcceptableFloat(fVector[i])) {
                continue;
            }
            sum += Math.pow(Math.abs(fVector[i]), p);
        }
        sum = (float) Math.pow(sum, 1. / p);
        return sum;
    }

    /**
     * @param Integer array.
     * @return Norm of the given array.
     */
    public float norm(int[] iVector) throws Exception {
        if (iVector == null) {
            return 0f;
        }
        float sum = 0f;
        for (int i = 0; i < Math.min(3, iVector.length); i++) {
            if (!DataMineConstants.isAcceptableFloat(iVector[i])) {
                continue;
            }
            sum += Math.pow(Math.abs(iVector[i]), p);
        }
        sum = (float) Math.pow(sum, 1. / p);
        return sum;
    }

    @Override
    public float dist(int[] arrFirst, int[] arrSecond) throws MetricException {
        DistanceMeasure.assertArrays(arrFirst, arrSecond);
        float sum = 0;
        for (int i = 0; i < Math.min(3, arrFirst.length); i++) {
            if (!DataMineConstants.isAcceptableFloat(arrFirst[i])
                    || !DataMineConstants.isAcceptableFloat(arrSecond[i])) {
                continue;
            }
            sum += Math.pow(Math.abs((float) (arrFirst[i] - arrSecond[i])), p);
        }
        sum = (float) Math.pow(sum, 1. / p);
        return sum;
    }

    /**
     * @param first Data instance.
     * @param second Data instance.
     * @return Distance.
     * @throws MetricException
     */
    public float dist(DataInstance first, DataInstance second)
            throws MetricException {
        DistanceMeasure.assertInstances(first, second);
        float sum = 0f;
        DistanceMeasure.assertInstances(first, second);
        for (int i = 0; i < Math.min(3, first.getNumFAtt()); i++) {
            if (!DataMineConstants.isAcceptableFloat(first.fAttr[i])
                    || !DataMineConstants.isAcceptableFloat(second.fAttr[i])) {
                continue;
            }
            sum += (float) Math.pow(Math.abs(
                    (float) (first.fAttr[i] - second.fAttr[i])), p);
        }
        for (int i = 0; i < Math.min(3, first.getNumIAtt()); i++) {
            if (!DataMineConstants.isAcceptableInt(first.iAttr[i])
                    || !DataMineConstants.isAcceptableInt(second.iAttr[i])) {
                continue;
            }
            sum += (float) Math.pow(
                    Math.abs((float) (first.iAttr[i] - second.iAttr[i])), p);
        }
        sum = (float) Math.pow(sum, 1. / p);
        return sum;
    }

    /**
     * @param instance Data instance.
     * @return The norm.
     */
    public float norm(DataInstance instance) throws MetricException {
        if (instance == null) {
            return 0;
        }
        float sum = 0;
        for (int i = 0; i < Math.min(3, instance.getNumFAtt()); i++) {
            if (!DataMineConstants.isAcceptableFloat(instance.fAttr[i])) {
                continue;
            }
            sum += Math.pow(Math.abs((float) instance.fAttr[i]), p);
        }
        for (int i = 0; i < Math.min(3, instance.getNumIAtt()); i++) {
            if (!DataMineConstants.isAcceptableInt(instance.iAttr[i])) {
                continue;
            }
            sum += Math.pow(Math.abs((float) instance.iAttr[i]), p);
        }
        sum = (float) Math.pow(sum, 1. / p);
        return sum;
    }
}
