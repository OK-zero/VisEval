# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from xml.dom import minidom


def is_group_children(node):
    return node.nodeType == node.ELEMENT_NODE and node.tagName == "g"


def surface_form_check(svg_string):
    """
    Check if the code has plotted visualization.
    """
    doc = minidom.parseString(svg_string)
    svg = doc.getElementsByTagName("svg")[0]

    children = list(filter(lambda node: is_group_children(node), svg.childNodes))
    if len(children) == 0:
        return False, f"Did not plot visualization.\n{children}"

    if len(children) == 1:
        children = list(
            filter(lambda node: is_group_children(node), children[0].childNodes)
        )
        if len(children) < 2:
            return False, f"Did not plot visualization.\n{children}"

    return True, "Plotted visualization."
