import 'package:flutter/material.dart';

class ExpandableItem extends StatefulWidget {
  final Widget header;
  final Widget content;
  final bool isExpanded;
  final Function(bool) onToggle;

  const ExpandableItem({
    super.key,
    required this.header,
    required this.content,
    required this.isExpanded,
    required this.onToggle,
  });

  @override
  _ExpandableItemState createState() => _ExpandableItemState();
}

class _ExpandableItemState extends State<ExpandableItem> {
  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8.0),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.2),
            spreadRadius: 2,
            blurRadius: 5,
          ),
        ],
      ),
      child: Column(
        children: [
          InkWell(
            onTap: () => widget.onToggle(!widget.isExpanded),
            child: widget.header,
          ),
          if (widget.isExpanded) widget.content,
        ],
      ),
    );
  }
}
