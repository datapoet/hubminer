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

/**
 * This utility class enables sorting the DataInstance objects by some external
 * meta-criterion. Of course, this could be done separately while keeping track
 * of the resulting index permutation and often is. However, this class gives an
 * additional option by embedding the instances into sortable objects.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class DataSortableInfo {

    public int originalIndex = -1;
    public DataInstance originalInstance = null;
    public Object sortableData = null;
    public double primitiveSortable = Double.MIN_VALUE;

    /**
     * @param originalIndex Index of the original DataInstance object.
     * @param originalInstance DataInstance object to sort by external criteria.
     * @param primitiveSortable Primitive sorting criterion.
     */
    public DataSortableInfo(int originalIndex, DataInstance originalInstance,
            double primitiveSortable) {
        this.originalIndex = originalIndex;
        this.originalInstance = originalInstance;
        this.primitiveSortable = primitiveSortable;
    }

    /**
     * @param originalIndex Index of the original DataInstance object.
     * @param originalInstance DataInstance object to sort by external criteria.
     * @param sortableData The external sorting criterion.
     */
    public DataSortableInfo(int originalIndex, DataInstance originalInstance,
            Object sortableData) {
        this.originalIndex = originalIndex;
        this.originalInstance = originalInstance;
        this.sortableData = sortableData;
    }
}