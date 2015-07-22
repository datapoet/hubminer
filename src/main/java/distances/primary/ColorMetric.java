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

import java.awt.Color;
import java.io.Serializable;

/**
 * This class implements the distance between color RGB channels.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class ColorMetric implements Serializable {
    
    private static final long serialVersionUID = 1L;

    /**
     * Distance between Color RGB channels.
     *
     * @param first Color.
     * @param second Color.
     * @return Sum of absolute RGB distances.
     */
    public float dist(Color first, Color second) {
        float d = 0f;
        d += Math.abs(first.getRed() - second.getRed());
        d += Math.abs(first.getGreen() - second.getGreen());
        d += Math.abs(first.getBlue() - second.getBlue());
        return d;
    }
}
