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
        hovertemplate: '<b>Year</b> %{x} <b>Value</b>: %{y:.4f} ',
        // text = ['{}'.format(i + 1928) for i in range(len(spend_df))],        
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
          hovermode: 'closest',
          yaxis: {title: 'Spending',
            linecolor: 'black',
            mirror: true,
            ticks: 'inside',
            range: [0, 20]
            // rangemode: 'tozero'
          },
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