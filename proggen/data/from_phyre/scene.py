#!/usr/bin/env python
# coding=utf-8

# Make a copy of scene.thrift in Python

import enum
from dataclasses import dataclass
from typing import Union, List, Optional

import dotmap

@dataclass
class Vector:
    x: float
    y: float

@dataclass
class IntVector:
    x: int
    y: int

BodyType = enum.Enum('BodyType', [('STATIC', 1), ('DYNAMIC', 2)])
ShapeType = enum.Enum('ShapeType', 'UNDEFINED BALL BAR JAR STANDINGSTICKS')
Color = enum.Enum('Color', [
    ('WHITE', 0),
    ('RED', 1),
    ('GREEN', 2),
    ('BLUE', 3),
    ('PURPLE', 4),
    ('GRAY', 5),
    ('BLACK', 6),
    ('LIGHT_RED', 7),
])

@dataclass
class Polygon:
    vertices: List[Vector]

@dataclass
class Circle:
    radius: float

@dataclass
class Shape:
    polygon: Optional[Polygon] = None
    circle: Optional[Circle] = None

@dataclass
class Body:
    position: Vector
    bodyType: BodyType
    angle: Optional[float] = 0.
    shapes: Optional[List[Shape]] = None
    color: Optional[Color] = None
    shapeType: Optional[ShapeType] = None
    diameter: Optional[float] = None

scene_if = dotmap.DotMap({
    'Vector': Vector,
    'IntVector': IntVector,
    'BodyType': BodyType,
    'ShapeType': ShapeType,
    'Color': Color,
    'Polygon': Polygon,
    'Circle': Circle,
    'Shape': Shape,
    'Body': Body,
}, _dynamic=False)


