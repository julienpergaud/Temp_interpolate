function createMap(id, capteurs) {
    let elemName = "map" + id.toString();
    var mymap = L.map(elemName).setView([47.3167, 5.0167], 13);
    L.tileLayer('https://a.tile.openstreetmap.org/{z}/{x}/{y}.png ', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap contributors</a>'
    }).addTo(mymap);

    var captMap = new Map(Object.entries(capteurs));
    var markersTab = [];

    for (var capt of captMap) {
        var _capt = new Map(Object.entries(capt[1]));
        if (_capt.get('id_network') === id) {
            console.log('test reussi');
            var x = _capt.get('latitude');
            var y = _capt.get('longitude');
            var marker = L.marker([x, y]).addTo(mymap);
            markersTab.push(marker);
            marker.bindPopup(" <div class=\"w3-card-4\">\n" +
                "\n" +
                "<header class=\"w3-container w3-blue\">\n" +
                "  <h2>" + _capt.get('name') + "</h2>\n" +
                "</header>\n" +
                "\n" +
                "<div class=\"w3-container\">\n" +
                "  <p><strong> Position : </strong>[" + x + "," + y + "]</p>\n" +
                "</div>\n" +
                "\n" +
                /*"<footer class=\"w3-container w3-center w3-padding\">\n" +
                "  <a href=" + Flask.url_for("network", {"id_network":_capt.get('id_network')}) + " class=\"w3-button w3-green\n" +
                "    w3-round-large style=\"width:150px\"> Détails du reseau de capteur</a>\n" +
                "</footer>\n" +
                "\n" +*/
                "</div> ")

        }
    }
    group = L.featureGroup(markersTab);
    mymap.fitBounds(group.getBounds());
}

blueIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-blue.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

goldIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-gold.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

greenIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-green.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

greyIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-grey.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

orangeIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-orange.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

redIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-red.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

purpleIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-violet.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

yellowIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-yellow.png',
    shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/0.7.7/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41]
});

tabIcons = [blueIcon, goldIcon, greyIcon, greenIcon, orangeIcon, redIcon, purpleIcon, yellowIcon];
tabColors = ["blue", "amber", "grey", "green", "orange", "red", "purple", "yellow"];

function createMapNetwork(variables) {
    let elemName = "mapSensor";
    console.log(variables)
    var mymap = L.map(elemName).setView([47.3167, 5.0167], 13);
    L.tileLayer('https://a.tile.openstreetmap.org/{z}/{x}/{y}.png ', {
        maxZoom: 19,
        attribution: '&copy; <a href="https://openstreetmap.org/copyright">OpenStreetMap contributors</a>'
    }).addTo(mymap);

    var variablesMap = new Map(Object.entries(variables));
    var markersTab = [];
    var i = 0

    console.log(variablesMap)
    for (var variable of variablesMap) {
        console.log(variable)
        var sensinfo = new Map(Object.entries(variable[1]));
        console.log(sensinfo);
       
        if (typeof sensinfo.get('latitude') == "undefined") {
            continue;
        }
        var x = sensinfo.get('latitude');
        var y = sensinfo.get('longitude');
        /**var marker = L.marker([x, y], {icon: tabIcons[i]}).addTo(mymap);**/
        console.log(tabIcons[0],i)
        var marker = L.marker([x, y], {icon: tabIcons[0]}).addTo(mymap);
        markersTab.push(marker);
        htmlstring = "<div class=\"w3-card-4\">\n" +
            "\n" +
            "<header class=\"w3-container w3-" + tabColors[0] + "\">\n" +
            "  <h2>" + sensinfo.get('name') + "</h2>\n" +
            "</header>\n" +
            "\n" +
            "<div class=\"w3-container\">\n" +
            "  <p><strong> Position : </strong>[" + x + "," + y + "]</p>\n"

            for (var i=0; i<sensinfo.get('paramsName').length;i++){
                htmlstring += "<p><strong>"+ sensinfo.get('paramsName')[i]+" : </strong> "+ sensinfo.get('paramsValue')[i]+"</p>\n"
            }
            htmlstring += "</div>\n" +
            "\n" +
            "</div> "
            marker.bindPopup(htmlstring);
    i++;
    }
    group = L.featureGroup(markersTab);
    mymap.fitBounds(group.getBounds());
}






function createTree(data,_id) {
    JSONdata = data[0]['json_agg'];

    $('#arbre'+(_id)).jstree({
        'core': {
            'data': JSONdata
        },
        'checkbox': {
            cascade: 'up',
            whole_node: false,
            tie_selection: false,
        },
        'plugins': ["checkbox", "changed"]
    });

}

function getinfo(id){

            selected_node = $('#arbre'+(id)).jstree("get_checked", true);
            selected_bottom_node = $('#arbre'+(id)).jstree("get_bottom_checked", true);
            var data = {
                idvar: [],
                idsensor: [],
                idnetwork: [],
                namenetwork: [],
                namesvar: [],
                namesensor: [],
            };

            for (var i = 0; i < selected_bottom_node.length; i++) {
                data['idvar'].push(selected_bottom_node[i].original['id_var']);
                data['namesvar'].push(selected_bottom_node[i].original['text']);
                for (var y = 0; y < selected_bottom_node[i].parents.length; y++) {
                    parents = $('#arbre'+(id)).jstree("get_node", selected_bottom_node[i].parents[y]);
                    if (parents.id != '#') {
                        if (parents.original['id_sensor'] != undefined) {
                            data['idsensor'].push(parents.original['id_sensor']);
                            data['namesensor'].push(parents.original['text']);
                        }
                        if (parents.original['id_network'] != undefined) {
                            data['idnetwork'].push(parents.original['id_network']);
                            data['namenetwork'].push(parents.original['text']);
                        }
                    }
                }
            }

            $.post('/ajaxpostmethod', {
                id: id,
                paramsList: JSON.stringify(data),
                dateStart: document.getElementById("dateDebut"+id).value,
                dateEnd: document.getElementById("dateFin"+id).value
            });
            setTimeout(function() {location.reload()},500)
}

function readyTree(data,id){
    createTree(data,id);
}

function createGraph(data,id) {
    console.log(data)
    if (data === 'vide'){
        return;
    }
    console.log(data);
    _title = "Graph : ";
    _series=[];
    for (var i = 0; i < data['idvar'].length; i++) {
        _series[i] = {};
        _series[i]['name'] = data['namesensor'][i].toString() +" ("+data['namenetwork'][i].toString() +") : "+ data['namesvar'][i].toString() + " (" + data['unite'][i].toString() +") " ;
        _series[i]['data'] = [];
        for(var y= 0; y<data['data'][i].length;y++){
            date = new Date(data['data'][i][y]['date']);
            _series[i]['data'].push([Date.UTC(date.getFullYear(),date.getMonth(),date.getDate(), date.getHours(), date.getMinutes(), date.getSeconds()), data['data'][i][y]['measure']]);
        }
    }

    Highcharts.chart('graph'+(id)
        , {
            title: {
                text: _title
            },

            subtitle: {
                text: ""
            },
            xAxis: {
                type: 'datetime',
                dateTimeLabelFormats: { // don't display the dummy year
                    month: '%e. %b',
                    year: '%y'
                },
                title: {
                    text: 'Date'
                }
            },
            yAxis: {
                title: {
                    text: 'measures'
                },
                min: 0
            },
            tooltip: {
                headerFormat: '{series.name}<br>',
                pointFormat: '({point.x:%e / %b / %y}) | <b>{point.y:.2f}</b>'
            },

            plotOptions: {
                series: {
                    marker: {
                        enabled: true
                    }
                }
            },

            colors: ['#6CF', '#39F', '#06C', '#036', '#000'],

            // Define the data points. All series have a dummy year
            // of 1970/71 in order to be compared on the same x axis. Note
            // that in JavaScript, months start at 0 for January, 1 for February etc.
            series: _series
            ,

            responsive: {
                rules: [{
                    condition: {
                        maxWidth: 500
                    },
                    chartOptions: {
                        plotOptions: {
                            series: {
                                marker: {
                                    radius: 2.5
                                }
                            }
                        }
                    }
                }]
            }
        });
}


function getCookie(name) {
    var v = document.cookie.match('(^|;) ?' + name + '=([^;]*)(;|$)');
    return v ? v[2] : null;
}

function setCookie(name, value, days) {
    var d = new Date;
    d.setTime(d.getTime() + 24*60*60*1000*days);
    document.cookie = name + "=" + value + ";path=/;expires=" + d.toGMTString();
}

function deleteCookie(name) { setCookie(name, '', -1); }


function delGraph(id){
        $('#graph'+(id))[0].innerHTML='';
        console.log($('#delButton'+(id))[0])
        $('#delButton'+(id)).hide()
        $('#expButton'+(id)).hide()
       deleteCookie("graph"+(id))
}
