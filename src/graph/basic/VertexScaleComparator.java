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
package graph.basic;

import java.util.Comparator;

/**
 * This class implements a comparator for sorting the graph vertices according
 * to their scale.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class VertexScaleComparator implements Comparator {

    public static final boolean DESCENDING = false;
    public static final boolean ASCENDING = true;
    private boolean mode = ASCENDING;

    /**
     * Initialization.
     *
     * @param mode Boolean flag indicating whether to use the ascending or the
     * descending mode.
     */
    public VertexScaleComparator(boolean mode) {
        this.mode = mode;
    }

    @Override
    public int compare(Object firstVertex, Object secondVertex) {
        if ((firstVertex != null) && (secondVertex != null)) {
            if (mode == ASCENDING) {
                return (int) (((VertexInstance) firstVertex).scale
                        - ((VertexInstance) secondVertex).scale);
            } else {
                return (int) (((VertexInstance) secondVertex).scale
                        - ((VertexInstance) firstVertex).scale);
            }
        } else {
            return 0;
        }
    }
}