import xml.etree.ElementTree as ET

def xmlfile2str(fname):
    tree = ET.parse(fname)
    return ET.tostring(tree.getroot(), encoding='unicode')

def remove_elem(parent, tag):
    for elem in parent.findall(tag):
        parent.remove(elem)
    for elem in parent:
        remove_elem(elem, tag)
def find_id(parent, id):
    for elem in parent:
        if elem.get('id') == id:
            return (parent, elem)
    for elem in parent:
        a = find_id(elem, id)
        if a is not None:
            return a
def replace_filename(elem, fname):
    for string in elem.findall('string'):
        if string.get('name') == 'filename':
            old_fname = string.attrib['value']
            string.attrib['value'] = fname
            return old_fname
def num_of(parent, tag):
    return len(parent.findall(tag))

def formatxml(parent: ET.Element, indent='\n'):
    children = list(parent)

    if len(children) > 0:
        parent.text = indent+'\t'
        for e in children:
            formatxml(e, indent+'\t')
        children[-1].tail = children[-1].tail[:-1]
    
    parent.tail = indent

def keep_sensors(scene_xml, sensor_ids):
    root = ET.XML(scene_xml)
    sensors = root.findall('sensor')
    for i, sensor in enumerate(sensors):
        if i not in sensor_ids:
            root.remove(sensor)
    return ET.tostring(root, encoding='unicode')

def preprocess_scene(fname):
    '''
    Construct source scene
    '''
    tree = ET.parse(fname); root = tree.getroot()
    p = find_id(root, 'target')
    if p is not None:
        p[0].remove(p[1])
    p = find_id(root, 'ref_mat')
    if p is not None:
        p[0].remove(p[1])
    source_scene = ET.tostring(root, encoding='unicode')

    '''
    Construct target scene
    '''
    tree = ET.parse(fname); root = tree.getroot()
    p = find_id(root, 'source')
    if p is not None:
        p[0].remove(p[1])
    p = find_id(root, 'opt_mat')
    if p is not None:
        p[0].remove(p[1])
    target_scene = ET.tostring(root, encoding='unicode')

    return {
        'src': source_scene,
        'tgt': target_scene,
        'n_sensors': num_of(root, 'sensor')
    }