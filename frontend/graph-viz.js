/**
 * D3 force-directed graph visualization.
 *
 * Renders both the static knowledge graph and the dynamic
 * session context graph. Nodes are colored by entity type
 * and sized by connection count. Supports zoom and drag.
 */

// ─── Color mapping by entity type ───────────────────

const TYPE_COLORS = {
    // Knowledge graph types
    plant: "#78716c",
    assembly_line: "#78716c",
    machine: "#7c3aed",
    sensor: "#3b82f6",
    plc: "#3b82f6",
    maintenance_event: "#f97316",
    technician: "#ec4899",
    operator: "#ec4899",
    part: "#eab308",
    supplier: "#06b6d4",
    supplier_batch: "#06b6d4",
    defect: "#dc2626",
    alert: "#dc2626",
    material: "#22c55e",
    process_log: "#78716c",
    // Context graph types
    context_entity: "#7c3aed",
    turn_marker: "#d4d4d4",
    reference: "#eab308",
};

/**
 * Render a force-directed graph into an SVG element.
 *
 * Clears any existing graph in the target SVG and draws
 * a new one from the provided nodes and links data.
 *
 * @param {string} selector - CSS selector for the SVG element.
 * @param {Array} nodes - Array of {id, name?, type} objects.
 * @param {Array} links - Array of {source, target, type?} objects.
 */
function renderForceGraph(selector, nodes, links) {
    const svg = d3.select(selector);
    svg.selectAll("*").remove();

    // Nothing to render
    if (!nodes || nodes.length === 0) {
        svg.append("text")
            .attr("x", "50%")
            .attr("y", "50%")
            .attr("text-anchor", "middle")
            .attr("fill", "#a8a29e")
            .attr("font-size", "13px")
            .text("No graph data. Chat first or click Load.");
        return;
    }

    const width = svg.node().clientWidth || 560;
    const height = svg.node().clientHeight || 450;

    // ── Build valid link references ──
    //
    // D3 force needs source/target to be node IDs that exist
    // in the nodes array. Filter out links to missing nodes.

    const nodeIds = new Set(nodes.map((n) => n.id));
    const validLinks = links.filter(
        (l) => nodeIds.has(l.source) && nodeIds.has(l.target)
    );

    // ── Count connections for node sizing ──

    const connectionCount = {};
    validLinks.forEach((l) => {
        connectionCount[l.source] = (connectionCount[l.source] || 0) + 1;
        connectionCount[l.target] = (connectionCount[l.target] || 0) + 1;
    });

    // ── Zoom container ──

    const g = svg.append("g");

    svg.call(
        d3.zoom()
            .scaleExtent([0.3, 4])
            .on("zoom", (event) => g.attr("transform", event.transform))
    );

    // ── Force simulation ──

    const simulation = d3
        .forceSimulation(nodes)
        .force(
            "link",
            d3.forceLink(validLinks)
                .id((d) => d.id)
                .distance(60)
        )
        .force("charge", d3.forceManyBody().strength(-120))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(20));

    // ── Draw links ──

    const link = g
        .append("g")
        .selectAll("line")
        .data(validLinks)
        .join("line")
        .attr("class", "link");

    // ── Draw nodes ──

    const node = g
        .append("g")
        .selectAll("g")
        .data(nodes)
        .join("g")
        .attr("class", "node")
        .call(drag(simulation));

    // Node circles
    node
        .append("circle")
        .attr("r", (d) => {
            const count = connectionCount[d.id] || 0;
            return Math.min(4 + count * 1.2, 14);
        })
        .attr("fill", (d) => TYPE_COLORS[d.type] || "#a8a29e")
        .attr("stroke", "#fff")
        .attr("stroke-width", 1.5);

    // Node labels
    node
        .append("text")
        .attr("dx", 12)
        .attr("dy", 3)
        .text((d) => d.name || d.id);

    // Tooltip on hover
    node.append("title").text((d) => {
        const type = d.type || "unknown";
        const name = d.name || d.id;
        return `${name}\nType: ${type}`;
    });

    // ── Tick update ──

    simulation.on("tick", () => {
        link
            .attr("x1", (d) => d.source.x)
            .attr("y1", (d) => d.source.y)
            .attr("x2", (d) => d.target.x)
            .attr("y2", (d) => d.target.y);

        node.attr("transform", (d) => `translate(${d.x},${d.y})`);
    });
}

/**
 * D3 drag behavior for nodes.
 *
 * Lets the user grab and reposition nodes. Temporarily
 * increases the simulation alpha to re-settle the layout.
 */
function drag(simulation) {
    return d3
        .drag()
        .on("start", (event, d) => {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        })
        .on("drag", (event, d) => {
            d.fx = event.x;
            d.fy = event.y;
        })
        .on("end", (event, d) => {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        });
}