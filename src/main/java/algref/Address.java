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
 * Objects of this class hold the address information.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Address {
    
    /**
     * Default constructor.
     */
    public Address() {
    }
    
    /**
     * Initialization.
     * 
     * @param country String representing the location country.
     * @param city String representing the location city.
     */
    public Address(String country, String city) {
        this.country = country;
        this.city = city;
    }
    
    private String country;
    private String city;
    private String streetName;
    private int number;

    /**
     * @return String representing the country.
     */
    public String getCountry() {
        return country;
    }

    /**
     * @param country String representing the country.
     */
    public void setCountry(String country) {
        this.country = country;
    }

    /**
     * @return String representing the city.
     */
    public String getCity() {
        return city;
    }

    /**
     * @param city String representing the city.
     */
    public void setCity(String city) {
        this.city = city;
    }

    /**
     * @return String representing the street name.
     */
    public String getStreetName() {
        return streetName;
    }

    /**
     * @param streetName String representing the street name.
     */
    public void setStreetName(String streetName) {
        this.streetName = streetName;
    }

    /**
     * @return int representing the street number.
     */
    public int getNumber() {
        return number;
    }

    /**
     * @param number int representing the street number.
     */
    public void setNumber(int number) {
        this.number = number;
    }
}
