// import { data } from 'jquery';
import React, { Component } from 'react';
import Plot from 'react-plotly.js';

class SwrPlotSpend extends Component {

  render() {

    let df = this.props.mainObj.state.portval_df;
    let n_cohorts = df.data.length;
    let plotly_data = []
    for(var i=0; i<n_cohorts; i++) {
      plotly_data.push({
        x: df.columns,
        y: df.data[i],
        name: df.index[i],
        type: 'scatter',
        mode: 'lines',
        color: df.data[i][df.data[i].length],
        line: {'width': 1},
        hovertemplate: '<b>Year</b> %{x} <b>Value</b>: %{y:.4f} ',
      }); 
    }
    return (
      <Plot
        data={plotly_data}
        layout={{width: 480,
          height: 300, 
          showlegend: false,
          title: false,
          hovermode: 'closest',
          yaxis: {title: 'Portfolio Value',
            linecolor: 'black',
            mirror: true,
            ticks: 'inside',
            range: [0,400]
            // rangemode: 'tozero'
          },
          xaxis: {title: 'Year',
            linecolor: 'black',
            mirror: true,
            ticks: 'inside'},
          margin: {
            t: 5,
            r: 10,
            l: 50,
            b: 40,
            pad: 0
          }}
        }
      />
    );
  }
}

export default SwrPlotSpend;