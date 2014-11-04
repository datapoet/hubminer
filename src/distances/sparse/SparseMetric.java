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
package distances.sparse;

import java.io.Serializable;
import java.util.HashMap;

/**
 * This class defines sparse metrics via the distance method on their respective
 * index to frequency maps.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class SparseMetric implements Serializable {
    
    private static final long serialVersionUID = 1L;

    /**
     * @param first First BoW map.
     * @param second Second BoW map.
     * @return Distance between the two BoW maps.
     * @throws Exception
     */
    public abstract float dist(HashMap<Integer, Float> first,
            HashMap<Integer, Float> second)
            throws Exception;
}
