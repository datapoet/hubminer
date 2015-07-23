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
package data.representation.util;

import data.representation.DataInstance;
import java.util.Comparator;

/**
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 * @extends java.util.Comparator
 */
public class DataInstanceDimComparator implements Comparator {

    int featureIndex = 0;
    int featureType = DataMineConstants.FLOAT;

    public DataInstanceDimComparator(int featureIndex, int featureType) {
        this.featureIndex = featureIndex;
        this.featureType = featureType;
    }

    @Override
    public int compare(Object first, Object second) {
        switch (featureType) {
            case DataMineConstants.FLOAT: {
                float tmpValue = 0;
                float firstValue = ((DataInstance) first).fAttr[featureIndex];
                float secondValue = ((DataInstance) second).fAttr[featureIndex];
                if (!DataMineConstants.isAcceptableFloat(firstValue)) {
                    firstValue = 0;
                }
                if (!DataMineConstants.isAcceptableFloat(secondValue)) {
                    secondValue = 0;
                }
                tmpValue = firstValue - secondValue;
                if (DataMineConstants.isZero(tmpValue)) {
                    return 0;
                } else if (tmpValue > 0) {
                    return 1;
                } else {
                    return -1;
                }
            }
            case DataMineConstants.INTEGER: {
                int tmpValue = 0;
                int firstValue = ((DataInstance) first).iAttr[featureIndex];
                int secondValue = ((DataInstance) second).iAttr[featureIndex];
                if (!DataMineConstants.isAcceptableInt(firstValue)) {
                    firstValue = 0;
                }
                if (!DataMineConstants.isAcceptableInt(secondValue)) {
                    secondValue = 0;
                }
                tmpValue = firstValue - secondValue;
                return tmpValue;
            }
            default:
                return 1;
        }
    }
}