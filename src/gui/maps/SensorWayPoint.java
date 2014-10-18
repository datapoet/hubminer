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
package gui.maps;

import org.jdesktop.swingx.mapviewer.Waypoint;

/**
 * This class corresponds to a sensor Waypoint on the map, with an associated
 * hubness and bad hubness value, calculated from the sensor streams beforehand.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class SensorWayPoint extends Waypoint {

    private float hubness;
    private float badHubness;

    /**
     * Initialization.
     *
     * @param x Float that is the X coordinate of this Waypoint on the map.
     * @param y Float that is the Y coordinate of this Waypoint on the map.
     * @param hubness Float that is the hubness of the measurement stream of the
     * corresponding sensor.
     * @param badHubness Float that is the bad hubness of the measurement stream
     * of the corresponding sensor.
     */
    public SensorWayPoint(float x, float y, float hubness, float badHubness) {
        super(x, y);
        this.hubness = hubness;
        this.badHubness = badHubness;
    }

    /**
     * @return Float that is the hubness of the measurement stream of this
     * sensor.
     */
    public float getHubness() {
        return hubness;
    }

    /**
     * @param hubness Float that is the hubness of the measurement stream of the
     * sensor.
     */
    public void setHubness(float hubness) {
        this.hubness = hubness;
    }

    /**
     * @return Float that is the bad hubness of the measurement stream of this
     * sensor.
     */
    public float getBadHubness() {
        return badHubness;
    }

    /**
     * @param badHubness Float that is the bad hubness of the measurement stream
     * of this sensor.
     */
    public void setBadHubness(float badHubness) {
        this.badHubness = badHubness;
    }
}
