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
package distances.kernel;

import com.google.gson.Gson;
import data.representation.DataInstance;
import data.representation.sparse.BOWInstance;
import java.util.HashMap;

/**
 * The class that all kernel functions should extend.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public abstract class Kernel {

    /**
     * The dot product in the mapped space.
     * 
     * @param x float[] that is the feature value array.
     * @param y float[] that is the feature value array.
     * @return float value that is the dot product in the mapped space.
     */
    public abstract float dot(float[] x, float[] y);

    /**
     * The dot product in the mapped space.
     * 
     * @param x HashMap<Integer, Float> representing sparse feature values.
     * @param y HashMap<Integer, Float> representing sparse feature values.
     * @return float value that is the dot product in the mapped space.
     */
    public abstract float dot(HashMap<Integer, Float> x,
            HashMap<Integer, Float> y);

    /**
     * The dot product in the mapped space.
     * 
     * @param firstInstance DataInstance that is the first instance.
     * @param secondInstance DataInstance that is the second instance.
     * @return float value that is the dot product in the mapped space.
     */
    public float dot(DataInstance firstInstance, DataInstance secondInstance) {
        if (firstInstance instanceof BOWInstance &&
                secondInstance instanceof BOWInstance) {
            return dot(((BOWInstance) firstInstance).getWordIndexesHash(),
                    ((BOWInstance) secondInstance).getWordIndexesHash());
        } else {
            return dot(firstInstance.fAttr, secondInstance.fAttr);
        }
    }
    
    @Override
    public String toString() {
        Gson gson = new Gson();
        StringBuilder sb = new StringBuilder();
        sb.append(this.getClass().getCanonicalName());
        sb.append(":");
        sb.append(gson.toJson(this, this.getClass()));
        return sb.toString();
    }
}
