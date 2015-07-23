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

import java.util.ArrayList;
import java.util.Calendar;

/**
 * This class is used for representing the data about a citable research
 * publication. This can be used to export the info into some commonly used
 * format, like bibtex. Algorithm implementations can implement methods that
 * return this object in order to be easily citable.
 *
 * @author Nenad Tomasev <nenad.tomasev at gmail.com>
 */
public class Publication {

    private ArrayList<Author> authors;
    private String title;
    private Publisher publisher;
    private String doi;
    private String url;
    private int startPage, endPage;
    private int year;
    public static final int YEAR_MIN = 1700;

    /**
     * Default constructor.
     */
    public Publication() {
    }

    /**
     * Initialization.
     *
     * @param title String that is the publication title.
     * @param authors ArrayList<Author> of publication authors.
     * @param publisher Publisher that published the publication.
     * @param doi String that is the DOI.
     * @param url String that is the URL.
     * @param startPage Integer that is the start page.
     * @param endPage Integer that is the end page.
     * @param year Integer that is the year of the publication.
     */
    public Publication(String title, ArrayList<Author> authors,
            Publisher publisher, String doi, String url, int startPage,
            int endPage, int year) {
        this.title = title;
        this.authors = authors;
        this.publisher = publisher;
        this.doi = doi;
        this.url = url;
        this.startPage = startPage;
        this.endPage = endPage;
        this.year = year;
        int currYear = Calendar.getInstance().get(Calendar.YEAR);
        if (year < YEAR_MIN || year > currYear) {
            throw new IllegalArgumentException("Incorrect year provided: "
                    + year);
        }
    }

    /**
     * @param authorIndex Integer that is the index of the author in the authors
     * list, starting from zero.
     * @return Author corresponding to the specified author index.
     */
    public Author getAuthor(int authorIndex) {
        if (authorIndex >= 0 && authorIndex < getNumAuthors()) {
            return authors.get(authorIndex);
        } else {
            return null;
        }
    }

    /**
     * @return The number of specified authors for this publication.
     */
    public int getNumAuthors() {
        if (hasAuthors()) {
            return authors.size();
        } else {
            return 0;
        }
    }

    /**
     * @return Boolean value that is true if the authors have been specified,
     * false otherwise.
     */
    public boolean hasAuthors() {
        return (authors != null && !authors.isEmpty());
    }

    /**
     * @param auth Author to add to the authors list.
     */
    public void addAuthor(Author auth) {
        if (authors == null) {
            authors = new ArrayList<>(4);
        }
        authors.add(auth);
    }

    /**
     * @return ArrayList<Author> that is the list of authors.
     */
    public ArrayList<Author> getAuthors() {
        return authors;
    }

    /**
     * @return String representing the title.
     */
    public String getTitle() {
        return title;
    }

    /**
     * @param title String representing the title.
     */
    public void setTitle(String title) {
        this.title = title;
    }

    /**
     * @return Publisher of the publication.
     */
    public Publisher getPublisher() {
        return publisher;
    }

    /**
     * @param publisher Publisher of the publication.
     */
    public void setPublisher(Publisher publisher) {
        this.publisher = publisher;
    }

    /**
     * @return Integer representing the start page number.
     */
    public int getStartPage() {
        return startPage;
    }

    /**
     * @param startPage Integer representing the start page number.
     */
    public void setStartPage(int startPage) {
        this.startPage = startPage;
    }

    /**
     * @return Integer representing the end page number.
     */
    public int getEndPage() {
        return endPage;
    }

    /**
     * @param endPage Integer representing the end page number.
     */
    public void setEndPage(int endPage) {
        this.endPage = endPage;
    }

    /**
     * @return Integer representing the year.
     */
    public int getYear() {
        return year;
    }

    /**
     * @param year Integer representing the year.
     */
    public void setYear(int year) {
        this.year = year;
    }

    /**
     * @return String representing the DOI.
     */
    public String getDoi() {
        return doi;
    }

    /**
     * @param doi String representing the DOI.
     */
    public void setDoi(String doi) {
        this.doi = doi;
    }

    /**
     * @return String representing the URL.
     */
    public String getUrl() {
        return url;
    }

    /**
     * @param url String representing the URL.
     */
    public void setUrl(String url) {
        this.url = url;
    }
}
