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

import java.util.Comparator;

public class DataSortableInfoComparator implements Comparator {

    public static final boolean SIMPLE = true;
    private boolean mode = SIMPLE;
    Comparator internalComparator = null;

    /**
     * @param mode Boolean flag indicating whether to sort by the meta-primitive
     * or the meta-object. True signifies the primitive double value.
     */
    public DataSortableInfoComparator(boolean mode) {
        this.mode = mode;
    }

    /**
     * @param comp Comparator for the meta-object.
     */
    public DataSortableInfoComparator(Comparator comp) {
        internalComparator = comp;
    }

    @Override
    public int compare(Object v1, Object v2) {
        if ((v1 != null) && (v2 != null)) {
            if (internalComparator != null) {
                return internalComparator.compare(
                        ((DataSortableInfo) v1).sortableData,
                        ((DataSortableInfo) v2).sortableData);
            } else {
                if (mode == SIMPLE) {
                    return (int) (((DataSortableInfo) v1).primitiveSortable
                            - ((DataSortableInfo) v2).primitiveSortable);
                } else {
                    return (int) (((DataSortableInfo) v2).primitiveSortable
                            - ((DataSortableInfo) v1).primitiveSortable);
                }
            }
        } else {
            return 0;
        }
    }
}