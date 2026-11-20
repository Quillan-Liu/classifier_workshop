"""将全部标签的模型分数绘制为无重叠的 SVG 气泡图。"""
from html import escape
from math import cos, sin, pi, hypot, sqrt
from random import random, shuffle


def probability_chart(items):
    """将空间节点投影到画布，以端点渐变线连接每一对平面标签圆。

    Args:
        items (list[dict]): 标签名称 ``class_name`` 及 [0, 1] 的 ``probability``。

    Returns:
        str: 可直接作为 image/svg+xml 响应发送的 SVG，文本经过 XML 转义。
    """
    ordered = sorted(items, key=lambda item: item['probability'], reverse=True)
    maximum = max((item['probability'] for item in ordered), default=0) or 1
    placed, groups, gradients, colors, depths = [], [], [], [], []
    distance = 0
    hue_offset = random() * 360
    hues = [(index * 360 / len(ordered) + hue_offset) % 360 for index in range(len(ordered))]
    shuffle(hues)
    for index, item in enumerate(ordered):
        strength = sqrt(item['probability'] / maximum)
        radius = 32 + 40 * strength
        x = y = 0
        depth = 0
        found = index == 0
        while not found:
            distance += 6
            for step in range(90):
                angle = step * 2 * pi / 90 + index * 2.4
                # 三维坐标绕镜头倾斜后透视投影，深度独立于分类概率。
                wx, wy, wz = cos(angle) * distance, sin(angle) * distance, sin(index * 2.4) * distance * .65
                camera_y = wy * cos(.4) - wz * sin(.4)
                camera_z = wy * sin(.4) + wz * cos(.4)
                perspective = 1 / (1 - camera_z / (distance * 3 + 600))
                x, y = wx * perspective, camera_y * perspective
                depth = camera_z / (distance + 1)
                previous_distance = hypot(*placed[-1][:2]) if placed else 0
                if (hypot(x, y) >= previous_distance and
                        all(hypot(x - px, y - py) >= radius + pr + 16 for px, py, pr in placed)):
                    found = True
                    break
        placed.append((x, y, radius))
        depths.append(depth)
        name = escape(item['class_name'])
        score = f"{item['probability'] * 100:.2f}%"
        color = '#fff' if strength > .65 else '#344650'
        size = min(14, radius * 1.65 / max(len(item['class_name']), 1))
        # 每次重新打乱配色，与概率排名无关，同时保持标签间的色相区分。
        hue = hues[index]
        lightness = 91 - strength * 49
        base_color = f'hsl({hue} 40% {lightness}%)'
        colors.append(base_color)
        rim_color = f'hsl({hue} 36% {min(96, lightness + 9)}%)'
        gradients.append(
            f'<radialGradient id="rim-{index}" gradientUnits="userSpaceOnUse" cx="{x}" cy="{y}" r="{radius + 1.1}">'
            f'<stop offset="{radius / (radius + 1.1)}" stop-color="{base_color}" stop-opacity=".2"/>'
            f'<stop offset="{(radius + .55) / (radius + 1.1)}" stop-color="{rim_color}" stop-opacity=".38"/>'
            f'<stop offset="1" stop-color="{rim_color}" stop-opacity=".55"/></radialGradient>'
        )
        # 圆本身保持纯色。描边及连线的远近变化表达深度，不模拟球面。
        groups.append(
            f'<g data-node-index="{index}" data-depth="{depth}"><title>{name} {score}</title>'
            f'<circle class="node-rim" cx="{x}" cy="{y}" r="{radius + .5}" '
            f'fill="none" stroke="url(#rim-{index})" stroke-width="1.2" filter="url(#liquid-rim)" aria-hidden="true"/>'
            f'<circle data-node-body="" cx="{x}" cy="{y}" r="{radius}" '
            f'fill="{base_color}" fill-opacity=".78"/>'
            f'<text x="{x}" y="{y - 5}" fill="{color}" font-size="{size}" font-weight="600">{name}</text>'
            f'<text x="{x}" y="{y + 17}" fill="{color}" font-size="12">{score}</text></g>'
        )
    edges = []
    # 全连接按无向图绘制，每对节点只有一条线，置于标签后方以免遮挡文字。
    for index, (x, y, radius) in enumerate(placed):
        for other in range(index):
            px, py, pr = placed[other]
            length = hypot(x - px, y - py)
            ux, uy = (px - x) / length, (py - y) / length
            x1, y1 = x + ux * radius, y + uy * radius
            x2, y2 = px - ux * pr, py - uy * pr
            edge_id = f'edge-{index}-{other}'
            near = (depths[index] + depths[other] + 2) / 4
            gradients.append(
                f'<linearGradient id="{edge_id}" gradientUnits="userSpaceOnUse" '
                f'x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
                f'<stop offset="0" stop-color="{colors[index]}" stop-opacity="{.4 + (depths[index] + 1) * .25}"/>'
                f'<stop offset="1" stop-color="{colors[other]}" stop-opacity="{.4 + (depths[other] + 1) * .25}"/></linearGradient>'
            )
            edges.append(f'<line data-source="{index}" data-target="{other}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                         f'stroke="url(#{edge_id})" stroke-width="{1.8 + near * 2.4}" stroke-opacity="{.3 + near * .5}"/>')
    extent = max([220] + [max(abs(x), abs(y)) + r + 30 for x, y, r in placed])
    # 位移与柔光只作用于外侧描边，填色和文字不参与液化。
    rim = ('<filter id="liquid-rim" x="-35%" y="-35%" width="170%" height="170%" color-interpolation-filters="sRGB">'
           '<feTurbulence type="fractalNoise" baseFrequency=".035" numOctaves="2" seed="7" result="noise"/>'
           '<feDisplacementMap in="SourceGraphic" in2="noise" scale="1.8" xChannelSelector="R" yChannelSelector="G" result="liquid"/>'
           '<feGaussianBlur in="liquid" stdDeviation=".9" result="glow"/>'
           '<feMerge><feMergeNode in="glow"/><feMergeNode in="glow"/><feMergeNode in="liquid"/></feMerge></filter>')
    motion = ('<style>.node-rim{transform-box:fill-box;transform-origin:center;animation:rim-flow 8s ease-in-out infinite}'
              '@keyframes rim-flow{0%,100%{transform:scale(1) rotate(-2deg);opacity:.55}'
              '50%{transform:scale(1.006) rotate(2deg);opacity:.9}}'
              '@media(prefers-reduced-motion:reduce){.node-rim{animation:none;opacity:.7}}</style>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{-extent} {-extent} {extent * 2} {extent * 2}" '
            'role="img" aria-label="标签概率分布" text-anchor="middle" '
            'font-family="Segoe UI,Microsoft YaHei,sans-serif"><defs>' + rim + ''.join(gradients)
            + '</defs>' + motion + '<g aria-hidden="true">' + ''.join(edges) + '</g>' + ''.join(groups) + '</svg>')
