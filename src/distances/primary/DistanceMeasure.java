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
import java.io.Serializable;

/**
 * Defines the metric interfaces.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DistanceMeasure implements Serializable {
    
    private static final long serialVersionUID = 1L;

    /**
     * @param arrFirst Integer array.
     * @param arrSecond Integer array.
     * @return Distance.
     * @throws Exception Dummy method.
     */
    public float dist(int[] arrFirst, int[] arrSecond) throws MetricException {
        return 0f;
    }

    /**
     * @param arrFirst Float array.
     * @param arrSecond Float array.
     * @return Distance.
     * @throws Exception Dummy method.
     */
    public float dist(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        return 0f;
    }

    /**
     * Throws a MetricException if some of the arrays are null or not equal in
     * length.
     *
     * @param arrFirst Float array.
     * @param arrSecond Float array.
     * @throws Exception
     */
    public static void assertArrays(float[] arrFirst, float[] arrSecond)
            throws MetricException {
        if ((arrFirst == null) || (arrSecond == null)) {
            throw new MetricException("Null feature array encountered.");
        }
        if (arrFirst.length != arrSecond.length) {
            throw new MetricException("Array length mismatch" + arrFirst.length
                    + "and" + arrSecond.length + ": fatal error");
        }
    }

    /**
     * Throws a MetricException if some of the arrays are null or not equal in
     * length.
     *
     * @param arrFirst Integer array.
     * @param arrSecond Integer array.
     * @throws Exception
     */
    public static void assertArrays(int[] arrFirst, int[] arrSecond)
            throws MetricException {
        if ((arrFirst == null) || (arrSecond == null)) {
            throw new MetricException("Null feature array encountered.");
        }
        if (arrFirst.length != arrSecond.length) {
            throw new MetricException("Array length mismatch" + arrFirst.length
                    + "and" + arrSecond.length + ": fatal error");
        }
    }

    /**
     * Throws a MetricException if some of the instances are null or there is a
     * mismatch in definitions.
     *
     * @param first DataInstance.
     * @param second DataInstance.
     * @throws Exception
     */
    public static void assertInstances(DataInstance first, DataInstance second)
            throws MetricException {
        if ((first == null) || (second == null)) {
            throw new MetricException("Null instance passed to metric.");
        }
        if (!first.getEmbeddingDataset().equalsInFeatureDefinition(
                second.getEmbeddingDataset())) {
            throw new MetricException("Dataset definitions mismatch.");
        }
    }
}