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
 * Objects of this class hold the data about the authors of publications and 
 * algorithms, where necessary.
 * 
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Author {
    
    // In case of multiple names, they should all be placed in the name string,
    // and abbreviated where appropriate. This is not done automatically here.
    private String name;
    private String surname;
    private String affiliation;
    private String email;
    
    /**
     * Default constructor.
     */
    public Author() {
    }
    
    /**
     * @param name String that is the author's name.
     * @param surname String that is the author's surname.
     */
    public Author(String name, String surname) {
        this.name = name;
        this.surname = surname;
    }

    /**
     * @return String that is the author's name.
     */
    public String getName() {
        return name;
    }

    /**
     * @param name String that is the author's name.
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * @return String that is the author's surname.
     */
    public String getSurname() {
        return surname;
    }

    /**
     * @param surname String that is the author's surname.
     */
    public void setSurname(String surname) {
        this.surname = surname;
    }

    /**
     * @return String that is the author's affiliation.
     */
    public String getAffiliation() {
        return affiliation;
    }

    /**
     * @param affiliation String that is the author's affiliation.
     */
    public void setAffiliation(String affiliation) {
        this.affiliation = affiliation;
    }

    /**
     * @return String that is the author's email.
     */
    public String getEmail() {
        return email;
    }

    /**
     * @param email String that is the author's email.
     */
    public void setEmail(String email) {
        this.email = email;
    }
}
