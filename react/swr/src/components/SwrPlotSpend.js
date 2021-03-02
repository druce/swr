// import { data } from 'jquery';
import React, { Component } from 'react';
import Plot from 'react-plotly.js';

class SwrPlotSpend extends Component {

  render() {

    let df = this.props.mainObj.state.spend_df;
    let n_cohorts = df.data.length;
    let plotly_data = []
    for(var i=0; i<n_cohorts; i++) {
      plotly_data.push({
        x: df.columns,
        y: df.data[i],
        name: df.index[i],
        type: 'scatter',
        mode: 'lines',
        line: {'width': 1},
      }); 
    }
    return (
      <Plot
        data={plotly_data}
        layout={{width: 480,
          height: 300, 
          showlegend: false,
          title: false,
          yaxis: {title: 'Spending',
            linecolor: 'black',
            mirror: true,
            ticks: 'inside'},
          xaxis: {title: 'Year',
            linecolor: 'black',
            mirror: true,
            ticks: 'inside'},
          margin: {
            t: 5,
            r: 10,
            l: 40,
            b: 40,
            pad: 0
          },
          autocolorscale: false,
          colorscale: 'Viridis',
        }
        }
      />
    );
  }
}

export default SwrPlotSpend;