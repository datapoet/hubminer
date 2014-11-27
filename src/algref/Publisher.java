/**
 * Hub Miner: a hubness-aware machine learning experimentation library.
 * Copyright (C) 2014 Nenad Tomasev. Email: nenad.tomasev at gmail.com
 * 
 * This program is free software: you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */
package algref;

/**
 * Objects of this class hold the published info.
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Publisher {
    
    /**
     * Default constructor.
     */
    public Publisher() {
    }
    
    /**
     * Initialization.
     * 
     * @param name String that is the publisher's name.
     * @param location String that is the publisher's location.
     */
    public Publisher(String name, Address location) {
        this.name = name;
        this.location = location;
    }
    
    private String name;
    private Address location;

    /**
     * @return String that is the publisher's name.
     */
    public String getName() {
        return name;
    }

    /**
     * @param name String that is the publisher's name.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * @return Address that is the publisher's location.
     */
    public Address getLocation() {
        return location;
    }

    /**
     * @param location Address that is the publisher's location.
     */
    public void setLocation(Address location) {
        this.location = location;
    }
}
