
tag_colors = {'genre': (0, .55, .27, 1), 'season':(.86, .08, .24, 1), 
            'decade':(.29, 0, .51, 1), 'budget': (.93, .68, .05, 1),
            'rating': (.96, .47, .35, 1), 'popularity': (.05, .31, .55, 1),
            'runtime':(.48, .4, .87, 1), 'revenue':(0, 0, 0, 1),
            '': (0, .77, .73, 1)}

def decade(year):
    if year < 1930:
        decade_tag = "Early 1900's"
    elif year >= 1930 and year < 1940:
        decade_tag = "30's"
    elif year >= 1940 and year < 1950:
        decade_tag = "40's"
    elif year >= 1950 and year < 1960:
        decade_tag = "50's"
    elif year >= 1960 and year < 1970:
        decade_tag = "60's"
    elif year >= 1970 and year < 1980:
        decade_tag = "70's"
    elif year >= 1980 and year < 1990:
        decade_tag = "80's"
    elif year >= 1990 and year < 2000:
        decade_tag = "90's"
    elif year >= 2000 and year <= 2010:
        decade_tag = "2000's"
    else:
        decade_tag="Recent"
    return decade_tag