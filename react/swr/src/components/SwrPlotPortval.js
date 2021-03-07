// import { data } from 'jquery';
import React, { Component } from 'react';
import Plot from 'react-plotly.js';
// import Plotly from 'plotly.js/dist/plotly'

class SwrPlotSpend extends Component {

  getcolor(val, low, high) {
    let palette = ['#b40426', '#b50927', '#b70d28', '#b8122a', '#ba162b', '#bb1b2c', '#bd1f2d', '#be242e', '#c0282f', '#c12b30', '#c32e31', '#c43032', '#c53334', '#c73635', '#c83836', '#ca3b37', '#cb3e38', '#cc403a', '#cd423b', '#cf453c', '#d0473d', '#d1493f', '#d24b40', '#d44e41', '#d55042', '#d65244', '#d75445', '#d85646', '#d95847', '#da5a49', '#dc5d4a', '#dd5f4b', '#de614d', '#df634e', '#e0654f', '#e16751', '#e26952', '#e36b54', '#e36c55', '#e46e56', '#e57058', '#e67259', '#e7745b', '#e8765c', '#e9785d', '#e97a5f', '#ea7b60', '#eb7d62', '#ec7f63', '#ec8165', '#ed8366', '#ee8468', '#ee8669', '#ef886b', '#f08a6c', '#f08b6e', '#f18d6f', '#f18f71', '#f29072', '#f29274', '#f39475', '#f39577', '#f39778', '#f4987a', '#f49a7b', '#f59c7d', '#f59d7e', '#f59f80', '#f5a081', '#f6a283', '#f6a385', '#f6a586', '#f7a688', '#f7a889', '#f7a98b', '#f7aa8c', '#f7ac8e', '#f7ad90', '#f7af91', '#f7b093', '#f7b194', '#f7b396', '#f7b497', '#f7b599', '#f7b79b', '#f7b89c', '#f7b99e', '#f7ba9f', '#f7bca1', '#f6bda2', '#f6bea4', '#f6bfa6', '#f5c0a7', '#f5c1a9', '#f5c2aa', '#f5c4ac', '#f4c5ad', '#f4c6af', '#f3c7b1', '#f3c8b2', '#f2c9b4', '#f2cab5', '#f2cbb7', '#f1ccb8', '#f1cdba', '#f0cdbb', '#efcebd', '#efcfbf', '#eed0c0', '#edd1c2', '#edd2c3', '#ecd3c5', '#ebd3c6', '#ead4c8', '#ead5c9', '#e9d5cb', '#e8d6cc', '#e7d7ce', '#e6d7cf', '#e5d8d1', '#e4d9d2', '#e3d9d3', '#e2dad5', '#e1dad6', '#e0dbd8', '#dfdbd9', '#dedcdb', '#dddcdc', '#dcdddd', '#dbdcde', '#dadce0', '#d9dce1', '#d8dce2', '#d7dce3', '#d6dce4', '#d5dbe5', '#d4dbe6', '#d3dbe7', '#d2dbe8', '#d1dae9', '#cfdaea', '#cedaeb', '#cdd9ec', '#ccd9ed', '#cbd8ee', '#cad8ef', '#c9d7f0', '#c7d7f0', '#c6d6f1', '#c5d6f2', '#c4d5f3', '#c3d5f4', '#c1d4f4', '#c0d4f5', '#bfd3f6', '#bed2f6', '#bcd2f7', '#bbd1f8', '#bad0f8', '#b9d0f9', '#b7cff9', '#b6cefa', '#b5cdfa', '#b3cdfb', '#b2ccfb', '#b1cbfc', '#afcafc', '#aec9fc', '#adc9fd', '#abc8fd', '#aac7fd', '#a9c6fd', '#a7c5fe', '#a6c4fe', '#a5c3fe', '#a3c2fe', '#a2c1ff', '#a1c0ff', '#9fbfff', '#9ebeff', '#9dbdff', '#9bbcff', '#9abbff', '#98b9ff', '#97b8ff', '#96b7ff', '#94b6ff', '#93b5fe', '#92b4fe', '#90b2fe', '#8fb1fe', '#8db0fe', '#8caffe', '#8badfd', '#89acfd', '#88abfd', '#86a9fc', '#85a8fc', '#84a7fc', '#82a6fb', '#81a4fb', '#80a3fa', '#7ea1fa', '#7da0f9', '#7b9ff9', '#7a9df8', '#799cf8', '#779af7', '#7699f6', '#7597f6', '#7396f5', '#7295f4', '#7093f3', '#6f92f3', '#6e90f2', '#6c8ff1', '#6b8df0', '#6a8bef', '#688aef', '#6788ee', '#6687ed', '#6485ec', '#6384eb', '#6282ea', '#6180e9', '#5f7fe8', '#5e7de7', '#5d7ce6', '#5b7ae5', '#5a78e4', '#5977e3', '#5875e1', '#5673e0', '#5572df', '#5470de', '#536edd', '#516ddb', '#506bda', '#4f69d9', '#4e68d8', '#4c66d6', '#4b64d5', '#4a63d3', '#4961d2', '#485fd1', '#465ecf', '#455cce', '#445acc', '#4358cb', '#4257c9', '#4055c8', '#3f53c6', '#3e51c5', '#3d50c3', '#3c4ec2', '#3b4cc0'];
    let fraction = (val - low) / (high - low);
    return palette[Math.round((palette.length - 1) * fraction)];
  }

  render() {

    let df = this.props.mainObj.state.portval_df;
    let n_cohorts = df.data.length;
    let plotly_data = []

    // find min and max ending values
    let lastvals = [];
    for(let i=0; i<n_cohorts; i++) {
      lastvals.push(df.data[i][df.data[i].length - 1]);
    }
    let last_low = Math.min.apply(Math, lastvals)
    let last_high = Math.max.apply(Math, lastvals)
    let line_options =  {}
    
    for(var i=0; i<n_cohorts; i++) {
      if (i===this.props.mainObj.state.highlight_index)
        line_options = {'width': 2, 'color': 'black'};
      else 
        line_options = {'width': 1, 'color': this.getcolor(df.data[i][df.data[i].length - 1], last_low, last_high)};

      plotly_data.push({
        x: df.columns,
        y: df.data[i],
        name: df.index[i],
        type: 'scatter',
        mode: 'lines',
        color: df.data[i][df.data[i].length],
        line: line_options,
        hovertemplate: '<b>Year</b> %{x} <b>Value</b>: %{y:.4f} ',
      });   
    }
    return (
      <Plot
      divId='portval_plot'
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
        onClick = {(arg) => {
          // var update = {
          //   opacity: 1,
          //   'line.width': 4,
          //   'line.color': 'black',
          // };
          // // console.log(arg.points[0].curveNumber);
          // Plotly.restyle('portval_plot', update, arg.points[0].curveNumber);
          this.props.mainObj.setState({highlight_index: arg.points[0].curveNumber});
        }}

      />
    );
  }
}

export default SwrPlotSpend;