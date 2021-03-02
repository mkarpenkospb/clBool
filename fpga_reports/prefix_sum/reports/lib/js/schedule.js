"use strict";

// Global variable:
// Commonly used by HLD report

var scheduleViewerColorTypeDictBeta = {
  'kernel': 'ggroupblack', 
  'bb': 'gtaskblue',
  'cluster': 'gtaskgreen',
  'inst': 'gtaskyellow',
  'speculation': 'gtaskred',
  'iteration': 'gtaskpurple'
};

var vScheduleGantt;  // gantt chart
var vScheduleFuncData;   // selected function data
var vBottleneckScheduleNodes = [];  // selected bottleneck schedule nodes

// ************* schdule viewer (beta) ***************************

/**
 * @function changeBarDetailsBeta callback to change details
 * @param {JSGantt.TaskItem} taskItem for changing the information in the details
 */
function changeBarDetailsBeta(taskItem) {
  if (taskItem instanceof JSGantt.TaskItem) {
    // We have to check innerHTML, as empty notes by default from jsgantt
    // is <span class="gTaskNotes"></span>
    var details = taskItem.getNotes().innerHTML;
    if (details) {
      document.getElementById("details").innerHTML = details;
    } else {
      clearDivContent();
    }
  }
}

function createItemFromNode(node, parentId, g) {
  // Create new bar
  let nodeColor = (scheduleViewerColorTypeDictBeta.hasOwnProperty(node.type)) ? scheduleViewerColorTypeDictBeta[node.type] : "ggroupblack";  
  let nodeDebug   = (hasDebug(node)) ? createDebugLocation(getFirstDebug(node)).outerHTML : " - ";
  let pGroup = "", pOpen = 1; 
  let nodeStart;
  let nodeEnd;
  let details = getDetails(node);
  let nodeDetails = "";
  if (details) 
    nodeDetails = getHTMLDetailsFromJSON(details, node['name']);

  // Some basic error checking and hack for now
  if (node.hasOwnProperty("start") && node.hasOwnProperty("end")) {
    if (isNaN(node.start) || isNaN(node.end)) {
      console.log("Error! Data error: start cycle or latency is not a number: " + node.name);
      return;  // don't add to avoid crashing
    }
    nodeStart = parseInt(node.start);
    nodeEnd = parseInt(node.end);

    if ( ! node.hasOwnProperty("children") ) {  // Only check for start and end when it is lowest hierarchy
      if (nodeStart < -1 || nodeEnd < -1) { //accept start and end =-1 (for top most parent)
        console.log("Error! Data error: negative start cycle or latency: " + node.name);
        return;  // don't add to avoid crashing
      }
      if (nodeStart === nodeEnd && nodeStart!= -1) {
        // Temporary workaround by setting the latency to 1/2 clock cycle
        nodeEnd = nodeStart + 1;
      }
    }
  } else {
    console.log("Warning! Node has no start and end " + node.name);
    return;  // don't add
  }

  pGroup = (node.hasOwnProperty("children") ? 1 : 0);  // Update pGroup=1 (collapsible) if node is a parent

  let vDepend = (node.hasOwnProperty('depend')) ? node.depend : '';

  // Parameters                    (  pID,    pName,      pStart,    pEnd,     pStyle,   pLink  pMile, pRes,       pComp,  pGroup,   pParent,  pOpen, pDepend, pCaption,  pNotes,   pGantt))
  g.AddTaskItem(new JSGantt.TaskItem(node.id, node.name, nodeStart, nodeEnd, nodeColor,  '',   0,     nodeDebug,  0,      pGroup,   parentId,  pOpen, vDepend,  '',       nodeDetails, g));
}

function flattenObjBeta(node, parentId, g) {
  createItemFromNode(node, parentId, g);

  // recursively call it's children
  if (node.hasOwnProperty("children")) {
    node['children'].forEach(function (childObj) {
      flattenObjBeta(childObj, node.id, g);  //node.id is parent ID of the child
    });
  }
}

function flattenScheduleJSONBeta(scheduleJSON, funcNode, g) {
  //create the function bar item
  let schFuncNode = { "name": funcNode[0].name,
                      "id": funcNode[0].id,
                      "start": -1,
                      "end": -1,
                      "type": funcNode[0].type,
                      "children": 1
                    };
  createItemFromNode(schFuncNode, 0, g);  // insert to the top most of the list, parentId =0

  scheduleJSON['nodes'].forEach(function (node) {
    flattenObjBeta(node, schFuncNode.id, g); 
  });
}

/**
 * @function renderSchedule clears everything that was there and restart the whole graph
 * 
 * @param {Object} scheduleDataJSON is a list of children nodes for a given C++ function level
 * @param {Integer} chartID the ID selected from tree.
 */
function renderScheduleBeta(scheduleDataJSON, chartID){  
  // error and update check
  if (chartID !== undefined && top_node_id === chartID) return;  // do nothing if user clicks the same ID
  top_node_id = chartID;

  // Clear details pane before rendering
  clearDivContent();

  let max_cycle = 0;
  let min_cycle = 0;
  // Format the data for the first time
  let CID = getViewerConst().gid;
  vScheduleGantt = new JSGantt.GanttChart(document.getElementById(CID), 1);
  var centerPaneWidth = (CID!=null && $('#'+CID).width()> 200? $('#'+CID).width(): 200);  // This is just an error handling, take width=200 if it's smaller than 200px

  // Render schedule viewer beta
  if (vScheduleGantt.getDivId() != null) {
    vScheduleGantt.setNotesCallback(changeBarDetailsBeta); // Set up callback for show details when bar is clicked
    vScheduleGantt.setShowRes(0);
    vScheduleGantt.setShowDebugLoc(1);
    vScheduleGantt.setShowDur(0);
    vScheduleGantt.setShowComp(0);
    vScheduleGantt.setShowStartDate(0);
    vScheduleGantt.setShowEndDate(0);
    vScheduleGantt.setShowStartCycle(0);
    vScheduleGantt.setShowEndCycle(0);
    vScheduleGantt.setCaptionType('Complete');  // Set to Show Caption (None,Caption,Resource,Duration,Complete)
    vScheduleGantt.setQuarterColWidth(36);
    vScheduleGantt.setShowTaskInfoLink(1); // Show link in tool tip (0/1)
    vScheduleGantt.setShowEndWeekDate(0); // Show/Hide the date for the last day of the week in header for daily view (1/0)
    vScheduleGantt.setUseSingleCell(10000); // Set the threshold at which we will only use one cell per table row (0 disables).  Helps with rendering performance for large charts.
    vScheduleGantt.setUseZoom(1);
    vScheduleGantt.setTotalHeight("100%");
    vScheduleGantt.setUseSort(0);
    vScheduleGantt.setLastHeadingRow(0);  // Flag to show/hide the minor heading at the last row

    var funcNode = $.grep(treeJSON.nodes, function(a) { return a.id == chartID; }); 
    flattenScheduleJSONBeta(scheduleDataJSON, funcNode, vScheduleGantt);

    scheduleDataJSON['nodes'].filter(function (node) {
      return node.type.indexOf("bb") >-1 ;
    }).forEach(function (node) {
      if (parseInt(node.end) > max_cycle) { max_cycle = parseInt(node.end); }
    });

    // Save the selected Function for bottleneck visualization later
    vScheduleFuncData = scheduleDataJSON;

    // Calculate total latency size base on min_cycle and max_cycle
    var latencySize = (max_cycle> min_cycle? max_cycle - min_cycle : 0 );
    if (latencySize <= 0) {
      $('#' + CID).html("Screen size too small to load data.");
      return;
    } else {
      vScheduleGantt.setFormatArr.apply(vScheduleGantt, createGanttZoomList(latencySize, vScheduleGantt.getCycleColWidth(), centerPaneWidth));
      vScheduleGantt.setFormat(vScheduleGantt.getFormatArr()[vScheduleGantt.getFormatArr().length-1]); //use the last element in zoomLevelAry to zoom in full
    }
    vScheduleGantt.Draw();
 
  } else {
    console.log("Warning! Fail initialize Gantt Chart.");
  }
  return;
}

function createGanttZoomList(latency, cycleColWidth, divWidth) {
  var zoomLevelList = new Array();
  var curZoom = 1;
  zoomLevelList.push(curZoom);

  var rightSpace = (divWidth/2);  // taking 50% of the parent's width (as minus the left panel, as well as paddings)
  var numOfCols = parseInt(rightSpace/cycleColWidth)-1;  // divide by clock cycle width, 1 clockCycleCol=18px

  if(latency> numOfCols) {
    var numOfLatencyPerCol = parseInt(latency/numOfCols);

    while(curZoom*2 < numOfLatencyPerCol) {
      curZoom *=2;
      zoomLevelList.push(curZoom);
    }
    zoomLevelList.push(numOfLatencyPerCol);  // for zoom-in-full, so that it could be nicely fit in right panel/space
  }
  return zoomLevelList;
}

function removeBottleneck(pSelBottleneck) {
  let vStartCycle = pSelBottleneck['start'];
  let vEndCycle = pSelBottleneck['end'];
  vScheduleGantt.removeVerticalLine(vStartCycle);
  vScheduleGantt.removeVerticalLine(vEndCycle);
  for (let ni=0; ni<pSelBottleneck['nodes'].length; ni++) {
    let vSelBottleneckNode = pSelBottleneck['nodes'][ni];
    if (vSelBottleneckNode.hasOwnProperty('depend')) {
      delete vSelBottleneckNode['depend'];
    }
  }
}

function addBottleneck(pBottleneck) {
  // Add veritial line if the schedule viewer contains the bottleneck
  // Error check: nothing is selected
  if (!vScheduleFuncData) return;

  // The format of the bottleneck is a list of nodes. Element 0 is the start and last 
  // element is the end. The start and end is relative to the block. The schedule
  // viewer uses timescale at the function scope.
  // Use the parent of start and end node to the scheduled cycle.
  let vBottleneckNodes = pBottleneck['nodes'];
  let vStartNode = vBottleneckNodes[0];
  let vStartLoopName = vStartNode['parent'];  // Loop header block name
  let vStartLoopRe = new RegExp(vStartLoopName);

  let vEndNode = vBottleneckNodes[vBottleneckNodes.length-1];
  let vEndLoopName = vEndNode['parent'];  // Loop latch block name
  let vEndLoopRe = new RegExp(vEndLoopName);

  let vLimiter = (pBottleneck['type'] === 'Occupancy limiter') ? true : false;

  let vStartCycle, vEndCycle;
  vStartCycle = vEndCycle = -1;
  
  let vNodeList = vScheduleFuncData['nodes'];
  for (let ni=0; ni<vNodeList.length; ni++) {
    // find the start and end cycle for the bottleneck
    if (vNodeList[ni]['name'].match(vStartLoopRe)) {
      vStartCycle = parseInt(vNodeList[ni]['start']) + Math.floor(parseFloat(vStartNode['start']));
    }
    if (vNodeList[ni]['name'].match(vEndLoopRe)) {
      vEndCycle = parseInt(vNodeList[ni]['start']) + Math.floor(parseFloat(vEndNode['end']));
      // TODO: we are missing one clock cycle for fMAX/II limiter
      vEndCycle = (vLimiter) ? vEndCycle : vEndCycle+1;
    }
    if (vStartCycle >= 0 && vEndCycle > 0) break;
  }

  // do nothing if this not the right function
  if (vStartCycle < 0 || vEndCycle < 0) return;

  let vScheduleLimiter = {
    'start': vStartCycle,
    'end': vEndCycle
  };

  // Always add the bottleneck with a pair of vertical lines
  vScheduleGantt.addVerticalLine(vStartCycle);
  vScheduleGantt.addVerticalLine(vEndCycle);

  // find the task id we need to add edges
  // TODO: fix the block ID's so bottleneck and schedule have the same ID
  let vSearchList = filterScheduleNode(vLimiter, pBottleneck['loop']);
  if (vLimiter) {
    // this is a workaround as the ID's for blocks in bottleneck.json
    // does not match with the ID's schedule.json

    let vBottleneckIDDict = {};  // schedule to bottleneck
    let vScheduleIDDict = {};  // bottleneck to schedule

    for (let ni=1; ni<vBottleneckNodes.length-1; ni++) {
      let vBottleneckNode = vBottleneckNodes[ni];
      for (let si=0; si<vSearchList.length; si++) {
        let vSearchNode = vSearchList[si];
        if (vSearchNode['name'] === vBottleneckNode['name']) {
          vBottleneckScheduleNodes.push(vSearchNode);
          vScheduleIDDict[vSearchNode['id']] = vBottleneckNode['id'];
          vBottleneckIDDict[vBottleneckNode['id']] = vSearchNode['id'];
          break;
        }
      }
    }

    let vBottleneckLinks = pBottleneck['links'];  // the links are dependency between the task item
    // foreach bottleneck nodes from schedule JSON
    // find any nodes in the list that have edges between them
    // workaround remap from bottleneck ID to schedule ID
    for (let ni=0; ni<vBottleneckScheduleNodes.length; ni++) {
      let vBottleneckSchNode = vBottleneckScheduleNodes[ni];
      let vBottleneckID = vScheduleIDDict[vBottleneckSchNode['id']];
      for (let li=0; li<vBottleneckLinks.length; li++) {
        // find the source and destination of that also happened to be in bottleneck schedule
        // JSON list
        let vBottleneckLink = vBottleneckLinks[li];
        let vBottleneckSrcID = vBottleneckLink['from'];
        let vBottleneckDestID = vBottleneckLink['to'];
        if (!vBottleneckLink.hasOwnProperty('reverse') && // ignore backedge
            vBottleneckSrcID !== vBottleneckDestID &&    // ignore self loop edge
            vBottleneckDestID === vBottleneckID &&
            vBottleneckIDDict.hasOwnProperty(vBottleneckSrcID)) {
          // convert the ID back to schedule json
          if (!vBottleneckSchNode.hasOwnProperty('depend')) {
            vBottleneckSchNode['depend'] = vBottleneckIDDict[vBottleneckSrcID].toString();
          } else {
            vBottleneckSchNode['depend'] = vBottleneckSchNode['depend'] + ',' + vBottleneckIDDict[vBottleneckSrcID].toString();
          }
        }
      }
    }
  }
  else {
    // find the list of nodes that nodes that needs to add edges
    let vBottleneckScheduleIDs = {};
    for (let ni=1; ni<vBottleneckNodes.length-1; ni++) {
      let vBottleneckNode = vBottleneckNodes[ni];
      for (let si=0; si<vSearchList.length; si++) {
        let vSearchNode = vSearchList[si];
        if (vSearchNode['type'] === 'cluster') {
          let vInstList = vSearchNode['children'];
          for (let i=0; i<vInstList.length; i++) {
            let vInstNode = vInstList[i];
            if (vInstNode['id'] === vBottleneckNode['id']) {
              vBottleneckScheduleNodes.push(vInstNode);
              vBottleneckScheduleIDs[vBottleneckNode['id']] = 1;
            }
          }
        } else if (vSearchNode['id'] === vBottleneckNode['id']) {
          vBottleneckScheduleNodes.push(vSearchNode);
          vBottleneckScheduleIDs[vBottleneckNode['id']] = 1;
        }
      }
    }

    let vBottleneckLinks = pBottleneck['links'];  // the links are dependency between the task item
    // foreach bottleneck nodes from schedule JSON
    // find any nodes in the list that have edges between them
    for (let ni=0; ni<vBottleneckScheduleNodes.length; ni++) {
      let vBottleneckSchNode = vBottleneckScheduleNodes[ni];
      let vBottleneckID = vBottleneckSchNode['id'];
      for (let li=0; li<vBottleneckLinks.length; li++) {
        // find the source and destination of that also happened to be in bottleneck schedule
        // JSON list
        let vBottleneckLink = vBottleneckLinks[li];
        let vBottleneckSrcID = vBottleneckLink['from'];
        let vBottleneckDestID = vBottleneckLink['to'];
        if (!vBottleneckLink.hasOwnProperty('reverse') && // ignore backedge
            vBottleneckSrcID !== vBottleneckDestID &&    // ignore self loop edge
            vBottleneckDestID === vBottleneckID &&
            vBottleneckScheduleIDs.hasOwnProperty(vBottleneckSrcID)) {
          // convert the ID back to schedule json
          if (!vBottleneckSchNode.hasOwnProperty('depend')) {
            vBottleneckSchNode['depend'] = vBottleneckSrcID.toString();
          } else {
            vBottleneckSchNode['depend'] = vBottleneckSchNode['depend'] + ',' + vBottleneckSrcID.toString();
          }
        }
      }
    }
  }
  vScheduleLimiter['nodes'] = vBottleneckScheduleNodes;
  return vScheduleLimiter;
}

function filterScheduleNode(pLimiter, pLoopName) {
  // if pLimiter, then it needs a list of block name
  let vNodeList = vScheduleFuncData['nodes'];
  if (pLimiter) {
    let vBlockList = [];
    for (let ni=0; ni<vNodeList.length; ni++) {
      vBlockList.push(vNodeList[ni]);
    }
    return vBlockList;
  }
  // first find the loop name and return list of instruction to avoid
  // duplicated names across blocks
  // TODO: fix bottleneck JSON loop name to user-friendly name
  let vLoopNameRe = new RegExp(pLoopName,"g");
  for (let ni=0; ni<vNodeList.length; ni++) {
    if (vNodeList[ni]['name'].match(vLoopNameRe)) {
      return vNodeList[ni]['children'];
    }
  }
  return [];
}

function redrawSchedule() {
  // clear everything and redraw all the node with edges
  vScheduleGantt.ClearTasks();
  vScheduleGantt.Draw();

  var funcNode = $.grep(treeJSON.nodes, function(a) { return a.id == top_node_id; }); 
  flattenScheduleJSONBeta(vScheduleFuncData, funcNode, vScheduleGantt);

  vScheduleGantt.Draw();
  vScheduleGantt.updateVerticalLines();
}